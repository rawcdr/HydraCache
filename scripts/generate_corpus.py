from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def generate_10k(year, r_d, net_sales, income, text_conflict=None):
    filename = f"data/raw/apple_10k_{year}.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Page 1
    c.drawString(100, 750, f"Apple Inc. Form 10-K for Fiscal Year {year}")
    c.drawString(100, 700, "Item 7. Management's Discussion and Analysis of Financial Condition")
    c.drawString(100, 680, f"Total net sales for {year} were ${net_sales} billion.")
    c.drawString(100, 660, f"Research and development (R&D) expense was ${r_d} billion.")
    c.drawString(100, 640, f"Net income was ${income} billion.")
    
    if text_conflict:
        c.drawString(100, 600, text_conflict)
        
    c.showPage()
    
    # Page 2 - Tables
    c.drawString(100, 750, "Financial Tables")
    c.drawString(100, 720, "Metric | Value (in billions)")
    c.drawString(100, 700, "---------------------------")
    c.drawString(100, 680, f"Net Sales | ${net_sales}")
    c.drawString(100, 660, f"R&D Expense | ${r_d}")
    c.drawString(100, 640, f"Net Income | ${income}")
    
    c.showPage()
    c.save()
    print(f"Created {filename}")

if __name__ == "__main__":
    generate_10k(
        2024, 
        r_d=30.0, 
        net_sales=390.0, 
        income=100.0, 
        text_conflict="In 2024, management restated 2023 net sales to $375 billion due to accounting changes."
    )
    
    generate_10k(
        2023, 
        r_d=28.0, 
        net_sales=383.0, 
        income=97.0, 
        text_conflict="Our reported net sales for 2023 are $383 billion under the previous standard."
    )
