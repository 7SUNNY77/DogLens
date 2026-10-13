"""DogLens Streamlit UI: upload a photo, select one dog, get the top-3 similar breeds.

Run: uv run streamlit run app.py
Only the model is cached process-wide; photos, ROI, results and feedback state live in the
per-session st.session_state and are never written to disk or logs.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from doglens import ui_state as ui
from doglens.app_texts import INTRO, LIMITATIONS, SCORE_CAPTION, VERDICT_LABELS
from doglens.catalog import CatalogEntry, load_catalog
from doglens.errors import DogLensError, RoiError
from doglens.feedback import DEFAULT_DB_PATH, FeedbackError, FeedbackStore
from doglens.predictor import Predictor
from doglens.roi import display_box_to_roi, make_preview, roi_to_display

BUNDLE_DIR = os.environ.get("DOGLENS_BUNDLE_DIR", "artifacts/model_bundle")
CATALOG_PATH = os.environ.get("DOGLENS_CATALOG", "assets/breeds.json")
FEEDBACK_DB = os.environ.get("DOGLENS_FEEDBACK_DB", str(DEFAULT_DB_PATH))
PREVIEW_MAX_SIDE = 640

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


@st.cache_resource(show_spinner="Загрузка модели…")
def load_predictor(bundle_dir: str, catalog_path: str) -> Predictor:
    """Shared, read-only model; no user data is ever cached."""
    return Predictor.from_bundle_dir(bundle_dir, catalog_path, "cpu")


def draw_regions(preview: Image.Image, geometry, user_roi, crop_box) -> Image.Image:
    canvas = preview.copy()
    draw = ImageDraw.Draw(canvas)
    width = max(2, round(max(canvas.size) / 200))
    if crop_box is not None:
        x0, y0, x1, y1 = roi_to_display(crop_box, geometry)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(30, 180, 60), width=width)
    if user_roi is not None:
        x0, y0, x1, y1 = roi_to_display(user_roi, geometry)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(230, 30, 30), width=width)
    return canvas


def catalog_entries(predictor: Predictor | None) -> list[CatalogEntry]:
    if predictor is not None:
        return predictor.catalog
    try:
        return load_catalog(CATALOG_PATH)[1]
    except DogLensError:
        return []


def render_catalog(entries: list[CatalogEntry]) -> None:
    st.subheader("Поддерживаемые породы")
    if not entries:
        st.warning("Каталог пород недоступен.")
        return
    query = st.text_input("Поиск породы", key="catalog_query", placeholder="например, terrier")
    q = (query or "").strip().lower()
    rows = [
        {"Название": e.display_name_ru, "English": e.name_en, "Ключ": e.breed_key}
        for e in entries
        if not q or q in e.display_name_ru.lower() or q in e.name_en.lower() or q in e.breed_key
    ]
    st.caption(
        f"Найдено: {len(rows)} из {len(entries)}. Русские названия пока не проверены — "
        "показываются английские."
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_result(state, predictor: Predictor) -> None:
    result = state["result"]
    st.subheader("Похожие породы")
    for rank, p in enumerate(result.predictions, start=1):
        name = p.name_ru if p.name_ru == p.name_en else f"{p.name_ru} ({p.name_en})"
        st.markdown(f"**{rank}. {name}** — оценка модели {p.score:.1%}")
        st.progress(min(max(p.score, 0.0), 1.0))
    st.caption(SCORE_CAPTION)
    for warning in result.warnings:
        st.warning(warning)
    with st.expander("Технические сведения"):
        st.json(
            {
                "model_version": result.model_version,
                "request_id": result.request_id,
                "user_roi": list(result.user_roi) if result.user_roi else "всё изображение",
                "crop_box": list(result.crop_box) if result.crop_box else None,
                "crop_policy": predictor.crop_policy(),
                "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
            }
        )
    render_feedback(state, predictor)


def render_feedback(state, predictor: Predictor) -> None:
    st.subheader("Оцените результат")
    st.caption("Сохраняются только версия модели, top-3 и ваш ответ. Фото не сохраняется.")
    verdict = st.radio(
        "Ваш ответ",
        list(VERDICT_LABELS),
        format_func=lambda v: VERDICT_LABELS[v],
        index=None,
        key=f"verdict_{state['submission_id']}",
    )
    entries = predictor.catalog
    breed = st.selectbox(
        "Порода, если вы её знаете (необязательно)",
        [e.breed_key for e in entries],
        index=None,
        format_func=lambda k: next(e.display_name_ru for e in entries if e.breed_key == k),
        key=f"breed_{state['submission_id']}",
    )
    if st.button("Отправить отзыв", disabled=verdict is None, key="send_feedback"):
        try:
            status = FeedbackStore(FEEDBACK_DB).submit(
                state["submission_id"],
                state["result"].model_version,
                state["result"].predictions,
                verdict,
                breed,
                known_breed_keys={e.breed_key for e in entries},
            )
            state["feedback_status"] = status
        except FeedbackError as exc:
            state["feedback_status"] = f"error:{exc.user_message}"
    status = state.get("feedback_status")
    if status == "created":
        st.success("Спасибо! Отзыв сохранён.")
    elif status == "duplicate":
        st.info("Отзыв для этого результата уже сохранён.")
    elif isinstance(status, str) and status.startswith("error:"):
        st.error(status.removeprefix("error:"))


def render_recognition(state, predictor: Predictor | None) -> None:
    uploaded = st.file_uploader(
        "Фотография собаки (JPEG, PNG или WebP, до 10 МиБ)",
        key=f"uploader_{state['uploader_nonce']}",
    )
    ui.handle_upload(state, uploaded.getvalue() if uploaded is not None else None)

    image = state.get("image")
    if image is None:
        if state.get("phase") == ui.Phase.ERROR and state.get("error"):
            st.error(state["error"])
        else:
            st.info("Загрузите фотографию, чтобы начать.")
        return

    preview, geometry = make_preview(image, PREVIEW_MAX_SIDE)
    dw, dh = geometry.display_size
    st.caption(f"Изображение {image.width}×{image.height} px (с учётом EXIF-ориентации)")
    whole = st.toggle(
        "Использовать всё изображение",
        value=bool(state.get("whole_image", True)),
        key=f"whole_{state['upload_digest'][:12]}",
    )
    roi = None
    if not whole:
        st.markdown("Выделите **одну** собаку: задайте рамку в координатах превью.")
        key = state["upload_digest"][:12]
        x0, x1 = st.slider("Рамка по горизонтали", 0, dw, (0, dw), key=f"x_{key}")
        y0, y1 = st.slider("Рамка по вертикали", 0, dh, (0, dh), key=f"y_{key}")
        try:
            roi = display_box_to_roi((x0, y0, x1, y1), geometry)
        except RoiError as exc:
            st.warning(exc.user_message)
    ui.set_region(state, whole, roi)

    crop = region = None
    if predictor is not None and (whole or roi is not None):
        crop, region = predictor.prepare(image, roi)
    left, right = st.columns(2)
    with left:
        st.image(
            draw_regions(preview, geometry, roi, region.crop_box if region else None),
            caption="Красная рамка — ваш выбор, зелёная — область с контекстом",
        )
    with right:
        if crop is not None:
            st.image(crop, caption=f"Будет распознано: {crop.width}×{crop.height} px")
            if not whole:
                st.caption("К вашей рамке добавлено по 10% контекста с каждой стороны.")

    if state.get("phase") == ui.Phase.ERROR:
        st.error(state.get("error") or ui.GENERIC_ERROR)
        ui.recover_from_error(state)

    busy = state.get("phase") == ui.Phase.PREDICTING
    if st.button(
        "Показать похожие породы",
        type="primary",
        disabled=predictor is None or busy or not ui.can_predict(state),
        key="predict",
    ):
        with st.spinner("Распознавание…"):
            ui.run_prediction(state, predictor.predict)
        if state.get("phase") == ui.Phase.ERROR:
            st.error(state.get("error") or ui.GENERIC_ERROR)
            ui.recover_from_error(state)

    if state.get("phase") == ui.Phase.RESULT and state.get("result") is not None:
        render_result(state, predictor)


def main() -> None:
    st.set_page_config(page_title="DogLens", layout="wide")
    state = st.session_state
    ui.init(state)
    st.title("DogLens — похожие породы собак")
    st.markdown(INTRO)

    predictor = None
    try:
        predictor = load_predictor(BUNDLE_DIR, CATALOG_PATH)
    except DogLensError as exc:
        st.error(exc.user_message)
        with st.expander("Подробности"):
            st.code(f"bundle: {BUNDLE_DIR}\n{exc.detail or exc.code}")

    tab_main, tab_catalog = st.tabs(["Распознавание", "Поддерживаемые породы"])
    with tab_main:
        render_recognition(state, predictor)
        if st.button("Очистить", key="clear"):
            ui.clear(state)
            st.rerun()
        with st.expander("Ограничения"):
            st.markdown(LIMITATIONS)
    with tab_catalog:
        render_catalog(catalog_entries(predictor))


main()
