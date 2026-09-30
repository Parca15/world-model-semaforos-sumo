// Genera el artículo completo (.docx) con los resultados de results/ y sus figuras.
//   cd articulo && npm install && node build.js ../Articulo_WorldModels_LSTM_TSMixer_Transformer_SUMO_final.docx
// Figuras: results/figures/ (pipeline y diagrama_arquitecturas.py) y articulo/img/ (diagramas del borrador).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, HeadingLevel,
  WidthType, BorderStyle, ShadingType, PageNumber, Footer, LevelFormat,
} = require("docx");

const IMG_DIRS = [path.join(__dirname, "img"), path.join(__dirname, "..", "results", "figures")];
const OUT = process.argv[2];
const FONT = "Times New Roman";
const PAGE_W = 11906, MARGIN = 1134;              // A4, márgenes de 2 cm
const CONTENT_W = PAGE_W - 2 * MARGIN;            // 9638 DXA
const PX_W = 620;                                 // ancho de figura en px (a 96 dpi ≈ 16,4 cm)

// ---------------------------------------------------------------- utilidades de texto
// Marcado mínimo en línea: **negrita**, *cursiva*.
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({
  children: runs(text), alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 276 }, ...opts,
});
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const EQ = (text) => new Paragraph({
  children: [new TextRun({ text, italics: true })], alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 },
});
const BULLET = (text) => new Paragraph({
  children: runs(text), numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 60, line: 276 },
});
const CAPTION = (label, text) => new Paragraph({
  children: [new TextRun({ text: label + " ", italics: true, bold: true, size: 20 }),
             new TextRun({ text, italics: true, size: 20 })],
  alignment: AlignmentType.JUSTIFIED, spacing: { before: 60, after: 240 },
});

// ---------------------------------------------------------------- figuras
let nFig = 0;
function figure(file, caption, widthPx = PX_W) {
  const buf = fs.readFileSync(IMG_DIRS.map((d) => path.join(d, file)).find((p) => fs.existsSync(p)));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);   // cabecera PNG
  const height = Math.round(widthPx * h / w);
  nFig += 1;
  return [
    new Paragraph({
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: widthPx, height },
                                altText: { title: `Figura ${nFig}`, description: caption, name: file } })],
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 0 }, keepNext: true,
    }),
    CAPTION(`Figura ${nFig}.`, caption),
  ];
}

// ---------------------------------------------------------------- tablas
let nTab = 0;
const border = { style: BorderStyle.SINGLE, size: 4, color: "8C8C8C" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths, caption, { fontSize = 18, firstColLeft = true } = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const w = widths.map((x) => Math.floor(x * scale));
  w[w.length - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  const cell = (text, i, header) => new TableCell({
    borders, width: { size: w[i], type: WidthType.DXA },
    shading: header ? { fill: "E8EEF6", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({
      children: runs(String(text), { size: fontSize, bold: header || undefined }),
      alignment: i === 0 && firstColLeft ? AlignmentType.LEFT : AlignmentType.CENTER,
    })],
  });
  nTab += 1;
  return [
    CAPTION(`Tabla ${nTab}.`, caption),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w,
      rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
             ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) }))],
    }),
    new Paragraph({ children: [], spacing: { after: 160 } }),
  ];
}

// ---------------------------------------------------------------- contenido
const C = require("./contenido.js")({ P, H1, H2, EQ, BULLET, figure, table, runs, Paragraph, TextRun, AlignmentType });

const doc = new Document({
  creator: "Grupo de investigación — Universidad del Quindío",
  title: "Modelos temporales para World Models aplicados al control inteligente de semáforos en SUMO",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT },
        paragraph: { spacing: { before: 300, after: 140 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, italics: true, font: FONT },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 },
                          margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: C,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("escrito", OUT, `${nFig} figuras, ${nTab} tablas`); });
