from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import BooleanObject, NameObject, IndirectObject, TextStringObject

def set_need_appearances_writer(writer: PdfWriter):
    try:
        catalog = writer._root_object
        if "/AcroForm" not in catalog:
            writer._root_object.update({NameObject("/AcroForm"): IndirectObject(len(writer._objects), 0, writer)})
        need_appearances = NameObject("/NeedAppearances")
        writer._root_object["/AcroForm"][need_appearances] = BooleanObject(True)
        return writer

    except Exception as e:
        print('set_need_appearances_writer() catch : ', repr(e))
        return writer

pdf_path = 'w2.pdf'

reader = PdfReader(pdf_path)
output = PdfWriter()
output = set_need_appearances_writer(output)

with open(pdf_path, 'rb') as f:
    pdf = PdfReader(f)
    page = pdf.pages[0]
    annotations = page['/Annots']
    for annotation in annotations:
        field = annotation.get_object()
        field_type = field.get('/FT')
        field_value = field.get('/V')
        if field_value is not None and field_type == '/Tx':
            if field_value.startswith('ph'):
                print (field)
                field.update({NameObject('/V'): TextStringObject("updated")})
                field.update({NameObject('/Q'): TextStringObject("0")})  # 0 for left alignment
                field.update({NameObject('/DA'): TextStringObject("/HelveticaLTStd 8.00 Tf 0.000 0.000 0.000 rg")})  # 0 for left alignment

                print(field.keys())
    output.add_page(page)

    with open('updated_form.pdf', 'wb') as output_file:
        output.write(output_file)