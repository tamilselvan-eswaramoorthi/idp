from fpdf import FPDF

def dataframe_to_pdf(df, output_pdf):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font('Arial', 'B', 12)
    
    # Adding column headers
    for col in df.columns:
        pdf.cell(40, 10, col, border=1)
    pdf.ln()

    # Adding data rows
    pdf.set_font('Arial', '', 12)
    for row in df.itertuples(index=False):
        for item in row:
            pdf.cell(40, 10, str(item), border=1)
        pdf.ln()

    pdf.output(output_pdf)