"""
Export raport PDF profesional cu rezumat analiza + metrici + episoade.
Foloseste reportlab pentru ca e inclus in requirements si mai stabil decat alternatives.
"""
from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                  TableStyle, PageBreak, KeepTogether)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# Culorile din UI, adaptate pentru print
COLOR_ACCENT = colors.HexColor('#0099cc')
COLOR_ALERT = colors.HexColor('#cc2255')
COLOR_OK = colors.HexColor('#008855')
COLOR_DARK = colors.HexColor('#1a1f2e')
COLOR_LIGHT = colors.HexColor('#f5f7fa')


def _format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f'{h:02d}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'


def generate_pdf_report(result, filename, ground_truth=None,
                         window_metrics=None, event_metrics=None):
    """
    Genereaza un raport PDF complet despre analiza unui EDF.

    Args:
        result: dict returnat de run_inference
        filename: numele fisierului EDF analizat
        ground_truth: optional, lista de (start, end)
        window_metrics: optional, dict cu metrici per-window
        event_metrics: optional, dict cu metrici per-event

    Returns:
        bytes cu PDF-ul generat
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f'Seizure Detection Report - {filename}',
        author='Seizure Detection System',
    )

    styles = getSampleStyleSheet()

    # Stiluri custom
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'],
        fontSize=20, textColor=COLOR_ACCENT,
        alignment=TA_LEFT, spaceAfter=6,
        fontName='Helvetica-Bold',
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#666'),
        alignment=TA_LEFT, spaceAfter=20,
    )
    h2_style = ParagraphStyle(
        'H2', parent=styles['Heading2'],
        fontSize=14, textColor=COLOR_DARK,
        spaceBefore=14, spaceAfter=8,
        fontName='Helvetica-Bold',
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, alignment=TA_LEFT,
        spaceAfter=6, leading=14,
    )
    small_style = ParagraphStyle(
        'Small', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#888'),
        alignment=TA_CENTER,
    )

    story = []

    # =========================================================================
    # HEADER
    # =========================================================================
    story.append(Paragraph('Raport analiza EEG', title_style))
    story.append(Paragraph(
        f'Fisier: <b>{filename}</b> | '
        f'Generat: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        subtitle_style
    ))

    # =========================================================================
    # INFORMATII FISIER
    # =========================================================================
    story.append(Paragraph('1. Informatii fisier', h2_style))

    fi = result['file_info']
    file_data = [
        ['Parametru', 'Valoare'],
        ['Durata totala', _format_time(fi['duration_sec'])],
        ['Frecventa esantionare', f"{fi['original_fs']} Hz" +
         (f" (resamplat la {fi['fs']} Hz)" if fi['was_resampled'] else '')],
        ['Tip montaj', fi['montage_type']],
        ['Canale detectate', f"{fi['n_channels_found']}/18"],
        ['Numar ferestre analizate', f"{result['inference']['n_windows']:,}"],
    ]
    if fi['missing_channels']:
        missing_str = ', '.join(fi['missing_channels'][:5])
        if len(fi['missing_channels']) > 5:
            missing_str += f' (+{len(fi["missing_channels"]) - 5})'
        file_data.append(['Canale lipsa', missing_str])

    tbl_file = Table(file_data, colWidths=[5 * cm, 11 * cm])
    tbl_file.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), COLOR_ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_LIGHT]),
    ]))
    story.append(tbl_file)

    # =========================================================================
    # CONFIGURATIE INFERENTA
    # =========================================================================
    story.append(Paragraph('2. Configuratie inferenta', h2_style))

    inf = result['inference']
    story.append(Paragraph(
        f"Pragul de decizie folosit: <b>{inf['threshold']:.3f}</b><br/>"
        f"Arhitectura model: Ensemble LightGBM + EEGNet v2<br/>"
        f"Smoothing: window size 15, minim 5 ferestre consecutive pentru alerta",
        body_style
    ))

    # =========================================================================
    # METRICI (daca avem ground truth)
    # =========================================================================
    if ground_truth and window_metrics and event_metrics:
        story.append(Paragraph('3. Metrici pe acest fisier', h2_style))

        story.append(Paragraph(
            f"Ground truth: <b>{len(ground_truth)} criza/crize</b> adnotate.",
            body_style
        ))

        # Tabel metrici per-window
        wm = window_metrics
        win_data = [
            ['Metrica', 'Valoare'],
            ['Sensibilitate', f"{wm['sensitivity']:.2%}"],
            ['Specificitate', f"{wm['specificity']:.2%}"],
            ['Precizie', f"{wm['precision']:.2%}"],
            ['F1 Score', f"{wm['f1']:.3f}"],
            ['False Positive Rate / ora', f"{wm['fpr_per_hour']:.2f}"],
            ['True Positives (ferestre)', f"{wm['tp']:,}"],
            ['False Positives', f"{wm['fp']:,}"],
            ['True Negatives', f"{wm['tn']:,}"],
            ['False Negatives', f"{wm['fn']:,}"],
        ]

        tbl_win = Table(win_data, colWidths=[8 * cm, 8 * cm])
        tbl_win.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_ACCENT),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_LIGHT]),
        ]))
        story.append(Paragraph('<b>Metrici per-window:</b>', body_style))
        story.append(tbl_win)

        # Metrici per-event
        em = event_metrics
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph('<b>Metrici per-event:</b>', body_style))
        evt_text = (
            f"Crize reale in fisier: <b>{em['n_gt_seizures']}</b><br/>"
            f"Episoade detectate: <b>{em['n_detected_episodes']}</b><br/>"
            f"Crize prinse (TP): <b>{em['tp']}/{em['n_gt_seizures']}</b>"
        )
        if em['sensitivity'] is not None:
            evt_text += f" = {em['sensitivity']:.2%} sensibilitate per-event<br/>"
        else:
            evt_text += '<br/>'
        evt_text += f"Crize ratate (FN): <b>{em['fn']}</b><br/>"
        evt_text += f"Alarme false (FP): <b>{em['fp']}</b>"
        if em['avg_latency_sec'] is not None:
            lat = em['avg_latency_sec']
            if lat > 0:
                evt_text += f"<br/>Latenta medie: <b>+{lat:.1f}s</b> (inainte de debut clinic)"
            else:
                evt_text += f"<br/>Latenta medie: <b>{-lat:.1f}s</b> (dupa debut)"
        story.append(Paragraph(evt_text, body_style))

        section_idx = 4
    else:
        section_idx = 3

    # =========================================================================
    # EPISOADE DETECTATE
    # =========================================================================
    episodes = result['episodes']
    story.append(Paragraph(f'{section_idx}. Episoade detectate', h2_style))

    if not episodes:
        story.append(Paragraph(
            '<i>Nu au fost detectate episoade la pragul curent.</i>',
            body_style
        ))
    else:
        story.append(Paragraph(
            f'Total: <b>{len(episodes)}</b> episoade detectate.', body_style
        ))

        # Construim tabelul
        headers = ['#', 'Start', 'Sfarsit', 'Durata', 'Prob. max', 'Nr. fer.']
        if ground_truth:
            headers.append('Status')

        eps_data = [headers]
        for i, ep in enumerate(episodes[:30], 1):  # limitam la primele 30
            row = [
                str(i),
                _format_time(ep['start_sec']),
                _format_time(ep['end_sec']),
                f"{ep['duration_sec']:.0f}s",
                f"{ep['max_prob']:.3f}",
                str(ep['n_windows']),
            ]
            if ground_truth:
                overlap = any(ep['start_sec'] < ge and ep['end_sec'] > gs
                              for gs, ge in ground_truth)
                row.append('TP' if overlap else 'FP')
            eps_data.append(row)

        if len(episodes) > 30:
            eps_data.append(['...'] * len(headers))

        col_widths = [1 * cm, 2.5 * cm, 2.5 * cm, 2 * cm, 2.5 * cm, 2 * cm]
        if ground_truth:
            col_widths.append(2 * cm)

        tbl_eps = Table(eps_data, colWidths=col_widths, repeatRows=1)
        style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_DARK),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLOR_LIGHT]),
        ]
        # Colorare status pe ultima coloana
        if ground_truth:
            for row_idx, row in enumerate(eps_data[1:], 1):
                if row[-1] == 'TP':
                    style_cmds.append(('TEXTCOLOR', (-1, row_idx), (-1, row_idx), COLOR_OK))
                elif row[-1] == 'FP':
                    style_cmds.append(('TEXTCOLOR', (-1, row_idx), (-1, row_idx), COLOR_ALERT))
        tbl_eps.setStyle(TableStyle(style_cmds))
        story.append(tbl_eps)

    # =========================================================================
    # INTERPRETARE
    # =========================================================================
    story.append(Paragraph(f'{section_idx + 1}. Interpretare', h2_style))

    interp_text = []

    if not episodes:
        interp_text.append(
            'Modelul nu a detectat episoade periculoase in acest fisier la pragul curent. '
            'Aceasta poate indica: (1) absenta reala a crizelor; (2) prag prea inalt '
            'pentru subtilitatea modificarilor din semnal; (3) caracteristici ale fisierului '
            'care nu se potrivesc cu distributia de antrenament.'
        )
    else:
        n_ep = len(episodes)
        total_dur = sum(ep['duration_sec'] for ep in episodes)
        pct_alert = 100 * total_dur / fi['duration_sec']

        interp_text.append(
            f'Modelul a detectat <b>{n_ep} episoade</b> cumuland <b>{total_dur:.0f} secunde</b> '
            f'({pct_alert:.1f}% din durata inregistrarii).'
        )

        if ground_truth and event_metrics:
            em = event_metrics
            if em['tp'] == em['n_gt_seizures']:
                interp_text.append(
                    'Toate crizele adnotate au fost detectate corect (100% sensibilitate per-event).'
                )
            elif em['tp'] > 0:
                interp_text.append(
                    f'Sistemul a prins <b>{em["tp"]} din {em["n_gt_seizures"]}</b> crize reale '
                    f'({em["sensitivity"]:.0%} sensibilitate per-event).'
                )
            else:
                interp_text.append(
                    'Nicio criza reala nu a fost detectata. Verifica adnotarile sau '
                    'incearca sa cobori pragul de decizie.'
                )

    for txt in interp_text:
        story.append(Paragraph(txt, body_style))

    # =========================================================================
    # FOOTER + NOTE METODOLOGICE
    # =========================================================================
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(f'{section_idx + 2}. Note metodologice', h2_style))

    notes = (
        '<b>Arhitectura:</b> Sistem hibrid cu doua modele complementare. '
        'LightGBM opereaza pe 194 features tabulare (band powers, wavelet, Hjorth, '
        'statistici, entropii), iar EEGNet (F1=32, D=6, F2=64) proceseaza direct semnalul brut. '
        'Predictiile sunt combinate ca medie ponderata 50/50.<br/><br/>'
        '<b>Dataset antrenament:</b> CHB-MIT Scalp EEG Database, 19 pacienti (chb06-chb24).<br/><br/>'
        '<b>Limitari:</b> Modelul a fost evaluat pe pacienti cunoscuti din dataset. '
        'Performanta pe pacienti complet noi poate sa difere. '
        'Sistemul nu substituie evaluarea clinica de specialitate si este destinat uzului '
        'academic si de cercetare.<br/><br/>'
        '<b>Unitate per-fereastra:</b> 4 secunde de semnal EEG, esantionat la 256 Hz.'
    )
    story.append(Paragraph(notes, body_style))

    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(
        'Generat de Seizure Detection System | Lucrare de licenta | '
        'Mierla Constantin | UBB Cluj-Napoca, 2026',
        small_style
    ))

    # Build
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
