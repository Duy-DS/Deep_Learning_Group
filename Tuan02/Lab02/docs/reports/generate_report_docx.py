import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_docx_report():
    doc = Document()
    
    # 1. Page Margins (1 inch = 2.54 cm)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.header_distance = Inches(0.5)
        section.footer_distance = Inches(0.5)

    # Base Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    normal_style.paragraph_format.line_spacing = 1.25
    normal_style.paragraph_format.space_after = Pt(6)

    # Helper functions
    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(20)
        run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79) # Navy
        p.paragraph_format.space_after = Pt(12)
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.italic = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(14)
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        p.paragraph_format.space_after = Pt(18)
        return p

    def add_heading_1(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(13.5)
        run.font.color.rgb = RGBColor(0x2F, 0x55, 0x97)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        return p

    def add_heading_3(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.italic = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(3)
        return p

    def add_bullet(text, level=0):
        p = doc.add_paragraph(style='List Bullet')
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        return p

    def add_callout(text, bold_prefix=""):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.right_indent = Inches(0.4)
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(8)
        if bold_prefix:
            r_b = p.add_run(bold_prefix + " ")
            r_b.bold = True
            r_b.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        r_t = p.add_run(text)
        r_t.italic = True
        return p

    def add_formula(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Cambria Math'
        run.font.size = Pt(12)
        run.bold = True
        run.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        return p

    def add_image_safe(img_path, caption, width=Inches(5.8)):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(8)
            p_img.paragraph_format.space_after = Pt(2)
            run = p_img.add_run()
            run.add_picture(img_path, width=width)
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_cap = p_cap.add_run(caption)
            r_cap.font.name = 'Times New Roman'
            r_cap.font.size = Pt(10.5)
            r_cap.italic = True
            r_cap.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
            p_cap.paragraph_format.space_after = Pt(10)

    # Base Paths
    root_dir = Path(__file__).resolve().parent.parent.parent
    figures_dir = root_dir / "outputs" / "figures"

    # --- TITLE SECTION ---
    add_title("BÁO CÁO KHOA HỌC: DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN AMES BẰNG MẠNG NƠ-RON MULTILAYER PERCEPTRON (PYTORCH)")
    add_subtitle("Theo Chuẩn Tiến Trình Khai Phá Dữ Liệu Công Nghiệp CRISP-DM\nCuộc thi Kaggle: House Prices - Advanced Regression Techniques")

    # Metadata Box Table
    meta_table = doc.add_table(rows=6, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Cơ quan đào tạo:", "[Tên Trường Đại học] – [Tên Khoa / Bộ môn]"),
        ("Học phần / Mã lớp:", "Deep Learning (Học sâu) – Mã lớp: [Điền Mã Lớp Học Phần]"),
        ("Giảng viên hướng dẫn:", "[Họ và tên Giảng viên hướng dẫn]"),
        ("Nhóm thực hiện:", "Nhóm 5 thành viên"),
        ("Nhóm trưởng:", "Nguyễn Tấn Duy – MSSV: 3123580008"),
        ("Thời gian thực hiện:", "Học kỳ [Điền Học Kỳ] – Năm học 2025–2026")
    ]
    set_table_borders(meta_table, color="B0C4DE", sz="6")
    for i, (label, val) in enumerate(meta_data):
        row = meta_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.2)
        c1.width = Inches(4.3)
        set_cell_background(c0, "F2F5F9")
        set_cell_margins(c0, top=80, bottom=80, left=120, right=120)
        set_cell_margins(c1, top=80, bottom=80, left=120, right=120)
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(label)
        r0.bold = True
        r0.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        p1 = c1.paragraphs[0]
        p1.add_run(val)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # --- TEAM ASSIGNMENT TABLE ---
    add_heading_2("BẢNG PHÂN CÔNG NHIỆM VỤ VÀ ĐÓNG GÓP CỦA THÀNH VIÊN")
    team_table = doc.add_table(rows=6, cols=6)
    team_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["STT", "Họ và tên", "MSSV", "Vai trò chính", "Nhiệm vụ chi tiết phụ trách", "Đóng góp"]
    widths = [Inches(0.5), Inches(1.5), Inches(1.1), Inches(1.2), Inches(2.0), Inches(0.7)]
    
    hdr_row = team_table.rows[0]
    for j, h in enumerate(headers):
        cell = hdr_row.cells[j]
        cell.width = widths[j]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=100, bottom=100, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(10.5)

    team_rows = [
        ("1", "Nguyễn Tấn Duy", "3123580008", "Trưởng nhóm", "Quản lý dự án, Business Understanding, kiến trúc hệ thống, tổng hợp và chuẩn hóa báo cáo CRISP-DM", "100%"),
        ("2", "[Thành viên 2 - Họ tên]", "[MSSV 2]", "Thành viên", "Data Understanding, phân tích EDA, trực quan hóa phân phối target và tương quan", "100%"),
        ("3", "[Thành viên 3 - Họ tên]", "[MSSV 3]", "Thành viên", "Data Preparation, thiết kế pipeline tiền xử lý, kiểm soát rò rỉ dữ liệu, One-Hot Encoding", "100%"),
        ("4", "[Thành viên 4 - Họ tên]", "[MSSV 4]", "Thành viên", "PyTorch Modeling, thiết kế mạng MLPRegressor, huấn luyện với Early Stopping và Dropout", "100%"),
        ("5", "[Thành viên 5 - Họ tên]", "[MSSV 5]", "Thành viên", "Baseline & Evaluation, mô hình đối chuẩn (Linear, Ridge, RF), đánh giá định lượng và sinh submission", "100%")
    ]

    set_table_borders(team_table, color="D3D3D3", sz="4")
    for i, data in enumerate(team_rows):
        row = team_table.rows[i+1]
        bg = "F9FBFD" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(data):
            cell = row.cells[j]
            cell.width = widths[j]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            if j in [0, 2, 5]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(10)
            if j == 1 and i == 0:
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # --- ABSTRACT ---
    add_heading_2("TÓM TẮT NỘI DUNG (ABSTRACT)")
    doc.add_paragraph(
        "Báo cáo trình bày toàn diện quá trình nghiên cứu, mô hình hóa và thực nghiệm giải quyết bài toán dự đoán giá bán bất động sản tại Ames, Iowa (dựa trên cuộc thi Kaggle House Prices: Advanced Regression Techniques và bài báo khoa học của Dean De Cock, 2011). Toàn bộ nghiên cứu được tổ chức tuần tự theo chuẩn mực công nghiệp CRISP-DM (Cross-Industry Standard Process for Data Mining) gồm 6 giai đoạn: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation và Deployment."
    )
    doc.add_paragraph(
        "Điểm nhấn học thuật của nghiên cứu tập trung vào việc tự thiết kế và huấn luyện kiến trúc mạng nơ-ron truyền thẳng Multilayer Perceptron (MLP) thuần PyTorch tích hợp phi tuyến ReLU, kỹ thuật điều hòa Dropout, Weight Decay và Early Stopping. Đồng thời, báo cáo đối chiếu định lượng với các mô hình hồi quy truyền thống (Linear Regression, Ridge Regression, Random Forest) và cung cấp luận cứ khoa học giải thích nguyên nhân các mô hình tuyến tính chính quy hóa lại đạt hiệu năng cao hơn mạng nơ-ron sâu trên tập dữ liệu dạng bảng quy mô nhỏ."
    )

    # --- PHASE 1: BUSINESS UNDERSTANDING ---
    add_heading_1("GIAI ĐOẠN 1: BUSINESS UNDERSTANDING (HIỂU BÀI TOÁN NGHIỆP VỤ)")
    add_heading_2("1.1. Bối cảnh nghiên cứu và Nguồn gốc dữ liệu")
    doc.add_paragraph(
        "Thị trường bất động sản là một trong những lĩnh vực kinh tế có mức độ phức tạp cao, trong đó giá trị một căn nhà phụ thuộc đồng thời vào rất nhiều yếu tố: vị trí địa lý, diện tích sử dụng, chất lượng xây dựng tổng thể, năm hoàn thành, tiện ích đi kèm (tầng hầm, nhà để xe, hồ bơi), và các điều kiện kinh tế vi mô. Trong nhiều thập kỷ, bộ dữ liệu Boston Housing Data (Harrison & Rubinfeld, 1978) đã được sử dụng rộng rãi như bộ dữ liệu quy chuẩn để nghiên cứu các thuật toán hồi quy. Tuy nhiên, như giáo sư Dean De Cock (2011) đã phân tích trong bài báo 'Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project', bộ dữ liệu Boston đã trở nên lạc hậu, kích thước mẫu nhỏ (506 quan sát), biến số thô sơ và chứa một số yếu tố nhân khẩu học nhạy cảm."
    )
    doc.add_paragraph(
        "Bộ dữ liệu Ames Housing được Dean De Cock thu thập trực tiếp từ Văn phòng Thẩm định Giá Thành phố Ames (Ames City Assessor's Office) trong giai đoạn 2006 – 2010. Dữ liệu ghi nhận thông tin thực tế của 2.930 giao dịch nhà ở dân cư cá nhân với 79 biến mô tả chi tiết, phản ánh toàn diện các thuộc tính vật lý và vị trí của một căn nhà. Cuộc thi Kaggle House Prices: Advanced Regression Techniques chia tập dữ liệu này thành hai phần: tập huấn luyện công khai (train.csv) gồm 1.460 dòng quan sát kèm giá bán thực tế (SalePrice); và tập kiểm thử độc lập (test.csv) gồm 1.459 dòng quan sát bị ẩn giá bán nhằm đánh giá khả năng tổng quát hóa trên bảng xếp hạng (Leaderboard)."
    )

    add_heading_2("1.2. Mục tiêu nghiên cứu và Định nghĩa bài toán")
    add_bullet("Mục tiêu kỹ thuật (Machine Learning Goal): Xây dựng mô hình học có giám sát (Supervised Learning - Regression) có khả năng ánh xạ véc-tơ đặc trưng thuộc tính ngôi nhà x in R^D thành giá trị dự đoán y_hat xấp xỉ giá bán thực tế y = SalePrice.")
    add_bullet("Mục tiêu thực hành Deep Learning: Tự thiết kế, lập trình và tinh chỉnh một mạng nơ-ron Multilayer Perceptron (MLP) thuần PyTorch, hiểu rõ cấu trúc tensor, cơ chế lan truyền ngược (backpropagation), hàm kích hoạt, tối ưu hóa và chống hiện tượng quá khớp (overfitting), thay vì chỉ dựa vào các thư viện mô hình đóng gói cấp cao.")
    add_bullet("Mục tiêu quy chuẩn MLOps: Tổ chức dự án khoa học, khả năng tái lập thực nghiệm (reproducibility), cấu trúc module hóa, không rò rỉ dữ liệu và tuân thủ chặt chẽ tiến trình CRISP-DM.")

    add_heading_2("1.3. Tiêu chí đánh giá chất lượng mô hình (Evaluation Metric)")
    doc.add_paragraph(
        "Theo chuẩn mực của cuộc thi Kaggle, thước đo chính thức để xếp hạng mô hình là Root-Mean-Squared-Error (RMSE) tính trên logarit của giá bán:"
    )
    add_formula("log-RMSE = sqrt( (1 / N) * sum_{i=1}^{N} ( ln(y_hat_i) - ln(y_i) )^2 )")
    doc.add_paragraph(
        "Ý nghĩa khoa học của log-RMSE trong định giá tài sản: Trong bài toán bất động sản, việc sử dụng sai số tuyệt đối hoặc sai số bình phương trên giá gốc (RMSE thông thường) sẽ gây ra độ lệch lớn khi dự đoán các căn nhà giá trị cao (ví dụ: căn nhà 700.000 USD sai lệch 50.000 USD sẽ bị phạt nặng gấp nhiều lần so với căn nhà 100.000 USD sai lệch 20.000 USD, dù xét về tỷ lệ phần trăm thì sai số căn thứ hai nghiêm trọng hơn). Việc lấy logarit giúp chuyển đổi sai số tuyệt đối thành sai số tỷ lệ phần trăm tương đối (relative percentage error): ln(y_hat) - ln(y) = ln(y_hat / y) ~ (y_hat - y) / y. Do đó, một sai lệch 10% trên ngôi nhà giá rẻ và ngôi nhà đắt tiền sẽ bị phạt tương đương nhau, đảm bảo tính công bằng và ổn định về mặt kinh tế lượng."
    )

    # --- PHASE 2: DATA UNDERSTANDING ---
    add_heading_1("GIAI ĐOẠN 2: DATA UNDERSTANDING (KHÁM PHÁ & HIỂU DỮ LIỆU - EDA)")
    add_heading_2("2.1. Phân loại cấu trúc 79 biến đặc trưng")
    doc.add_paragraph(
        "Bộ dữ liệu gồm 79 biến đầu vào mô tả 1.460 quan sát trong tập train.csv. Các đặc trưng được chia thành 4 nhóm hình thức dữ liệu chính:"
    )
    add_bullet("Biến số thực liên tục (Continuous Numerical - 20 biến): Thể hiện kích thước diện tích (LotFrontage, LotArea, MasVnrArea, BsmtFinSF1, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GrLivArea, GarageArea, WoodDeckSF, OpenPorchSF, v.v.).")
    add_bullet("Biến số nguyên rời rạc (Discrete Numerical - 16 biến): Thể hiện số lượng phòng và mốc thời gian (YearBuilt, YearRemodAdd, BsmtFullBath, FullBath, HalfBath, BedroomAbvGr, TotRmsAbvGrd, Fireplaces, GarageYrBlt, GarageCars, MoSold, YrSold, v.v.).")
    add_bullet("Biến định danh/Phân loại không thứ tự (Nominal Categorical - 23 biến): Loại hình bất động sản, vị trí và vật liệu (MSSubClass, MSZoning, Neighborhood, Condition1, BldgType, HouseStyle, RoofStyle, Foundation, Exterior1st, SaleType, SaleCondition, v.v.).")
    add_bullet("Biến thứ bậc/Phân loại có thứ tự (Ordinal Categorical - 20 biến): Đánh giá chất lượng và tiện nghi (OverallQual, OverallCond, ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, FireplaceQu, GarageQual, v.v.).")

    add_heading_2("2.2. Phân tích phân phối của biến mục tiêu SalePrice")
    doc.add_paragraph(
        "Phân tích thống kê mô tả cho biến SalePrice trên 1.460 căn nhà tập train ghi nhận: Giá trị nhỏ nhất Min = 34.900 USD, Trung vị Median = 163.000 USD, Trung bình Mean = 180.921 USD, và Giá trị lớn nhất Max = 755.000 USD. Độ lệch Skewness đạt +1.8828 (Lệch phải mạnh) và Độ nhọn Kurtosis đạt 6.5362 (Đuôi nặng - Leptokurtic)."
    )
    doc.add_paragraph(
        "Do giá bán có xu hướng phân phối lệch phải với đuôi kéo dài về phía giá cao, nếu huấn luyện trực tiếp trên thang đo gốc, mạng nơ-ron sẽ bị chi phối bởi các mẫu giá cao hiếm hoi, dẫn đến gradient bùng nổ và hiệu năng dự đoán kém ở vùng giá trung bình và thấp. Nghiên cứu áp dụng phép biến đổi logarit tự nhiên:"
    )
    add_formula("y_trans = ln(1 + SalePrice)")
    doc.add_paragraph(
        "Sau phép biến đổi, hệ số Skewness giảm mạnh xuống còn +0.1213, phân phối trở về dạng phân phối chuẩn (Gaussian-like distribution), hoàn toàn thỏa mãn các giả định hồi quy tuyến tính và giúp quá trình lan truyền gradient trong mạng MLP ổn định."
    )
    add_image_safe(str(figures_dir / "eda_target_distribution.png"), "Hình 1: Phân phối của biến mục tiêu SalePrice trước (lệch phải) và sau khi áp dụng phép biến đổi log1p (phân phối chuẩn).")

    add_heading_2("2.3. Phân tích cơ chế dữ liệu khuyết thiếu (Missing Data Mechanisms)")
    doc.add_paragraph(
        "Kiểm toán dữ liệu khuyết thiếu chỉ ra rằng có 19 cột chứa giá trị thiếu trên tập huấn luyện: PoolQC (99.52%), MiscFeature (96.30%), Alley (93.77%), Fence (80.75%), FireplaceQu (47.26%). Các nhóm đặc trưng liên quan đến Tầng hầm (Basement) và Nhà để xe (Garage) khuyết thiếu đồng thời từ 2.5% đến 5.5%."
    )
    add_callout(
        "Phần lớn các trường hợp khuyết thiếu trong bộ dữ liệu này không phải là dữ liệu thiếu ngẫu nhiên (Missing Completely at Random - MCAR) mà là dữ liệu thiếu có chủ đích mang ý nghĩa ngữ cảnh (Missing Not at Random - MNAR). Ví dụ: NA trong PoolQC có nghĩa là 'Ngôi nhà không có hồ bơi'; NA trong GarageType có nghĩa là 'Nhà không có garage'. Nếu xóa bỏ các hàng/cột này hoặc thay bằng mode phổ biến nhất, mô hình sẽ mất đi thông tin quý giá về sự thiếu vắng của tiện nghi.",
        bold_prefix="Insight Ngữ cảnh (Domain Knowledge):"
    )
    add_image_safe(str(figures_dir / "eda_missing_rates.png"), "Hình 2: Tỷ lệ khuyết thiếu của các đặc trưng trong tập huấn luyện (chủ yếu thuộc nhóm tiện ích đi kèm).")

    add_heading_2("2.4. Phân tích tương quan và Điểm ngoại lai (Outliers)")
    doc.add_paragraph(
        "Phân tích tương quan Pearson xác định các biến có mối quan hệ tuyến tính mạnh nhất với log(SalePrice): OverallQual (r = 0.8179), GrLivArea (r = 0.7009), GarageCars (r = 0.6806), GarageArea (r = 0.6509), TotalBsmtSF (r = 0.6121), và YearBuilt (r = 0.5866)."
    )
    add_image_safe(str(figures_dir / "eda_target_correlations.png"), "Hình 3: Ma trận tương quan giữa các đặc trưng số học hàng đầu và biến mục tiêu log(SalePrice).")
    doc.add_paragraph(
        "Dean De Cock (2011) chỉ ra sự hiện diện của các điểm ngoại lai khi đối chiếu GrLivArea với SalePrice: có hai căn nhà có diện tích sinh hoạt cực lớn (> 4.000 sq ft) nhưng giá bán lại rất thấp (< 300.000 USD), phản ánh các giao dịch nông nghiệp hoặc chuyển nhượng nội bộ đặc biệt cần được lưu ý khi đánh giá mô hình."
    )
    add_image_safe(str(figures_dir / "eda_grlivarea_outliers.png"), "Hình 4: Tương quan giữa diện tích GrLivArea và SalePrice cùng các điểm ngoại lai theo khuyến nghị của De Cock.")

    # --- PHASE 3: DATA PREPARATION ---
    add_heading_1("GIAI ĐOẠN 3: DATA PREPARATION (CHUẨN BỊ & TIỀN XỬ LÝ DỮ LIỆU)")
    doc.add_paragraph(
        "Để đảm bảo mô hình học tập hiệu quả và có khả năng áp dụng thực tế trên dữ liệu kiểm thử, quy trình tiền xử lý được thiết kế khép kín theo dạng pipeline với nguyên tắc Zero Data Leakage (mọi tham số thống kê chỉ được học trên tập Train, sau đó áp dụng biến đổi cho tập Validation và Test)."
    )

    add_heading_2("3.1. Chiến lược Điền giá trị khuyết thiếu theo Ngữ cảnh")
    add_bullet("Nhóm phân loại tiện nghi: Điền chuỗi 'None' cho các biến PoolQC, MiscFeature, Alley, Fence, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2 để biểu thị căn nhà không có tiện ích đó.")
    add_bullet("Nhóm kích thước tiện nghi liên quan: Điền giá trị 0 cho các biến diện tích/số lượng tương ứng khi nhà không có garage hoặc tầng hầm (GarageArea, GarageCars, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath, MasVnrArea).")
    add_bullet("Biến LotFrontage (Khoảng cách mặt đường): Điền bằng giá trị trung vị (median) của LotFrontage theo từng khu vực lân cận (Neighborhood).")
    add_bullet("Các biến số và biến phân loại còn lại: Điền bằng SimpleImputer(strategy='median') cho biến số và SimpleImputer(strategy='most_frequent') cho biến hạng mục.")

    add_heading_2("3.2. Mã hóa One-Hot và Chuẩn hóa Đặc trưng")
    doc.add_paragraph(
        "Mã hóa biến phân loại: Áp dụng OneHotEncoder(handle_unknown='ignore', sparse_output=False). Thuộc tính handle_unknown='ignore' đảm bảo khi tập kiểm thử xuất hiện các nhãn hiếm chưa từng có ở tập train, encoder sẽ tự động gán giá trị 0 cho các cột tương ứng thay vì báo lỗi kích thước. Không gian dữ liệu sau mã hóa mở rộng thành 318 đặc trưng số học."
    )
    doc.add_paragraph(
        "Chuẩn hóa thang đo (Feature Scaling): Do mạng nơ-ron học sâu tối ưu bằng gradient, sự chênh lệch thang đo giữa các biến (ví dụ diện tích hàng nghìn sq ft so với số phòng tắm 1-3) sẽ làm bề mặt hàm lỗi bị kéo dài, gây dao động gradient. Toàn bộ 318 đặc trưng được chuẩn hóa theo phương pháp Z-score Standardization:"
    )
    add_formula("x_scaled = (x - mu_train) / sigma_train")

    add_heading_2("3.3. Phân chia tập dữ liệu và Đóng gói Pipeline")
    doc.add_paragraph(
        "Tập dữ liệu huấn luyện (1.460 dòng) được phân tách thành hai tập: Train set (80% - 1.168 mẫu) và Validation set (20% - 292 mẫu), cố định random_state = 42. Pipeline tiền xử lý hoàn chỉnh được lưu trữ thành tệp nhị phân outputs/preprocessing/preprocessing_pipeline.pkl để phục vụ việc suy diễn nhất quán trên tập kiểm thử test.csv."
    )

    # --- PHASE 4: MODELING ---
    add_heading_1("GIAI ĐOẠN 4: MODELING (XÂY DỰNG MÔ HÌNH HỌC SÂU & ĐỐI CHUẨN)")
    add_heading_2("4.1. Cơ sở lý thuyết Mạng Nơ-ron Multilayer Perceptron (MLP)")
    doc.add_paragraph(
        "Mạng nơ-ron truyền thẳng đa tầng (MLP) là một trong những cấu trúc nền tảng của học sâu, hoạt động như một bộ xấp xỉ hàm vạn năng (Universal Function Approximator). Kiến trúc mạng được xây dựng gồm 3 tầng kết nối đầy đủ (Fully Connected Layers):"
    )
    add_callout(
        "Input (318 features) -> [Linear: 318 -> 128] -> [ReLU] -> [Dropout (p=0.2)] -> [Linear: 128 -> 64] -> [ReLU] -> [Dropout (p=0.2)] -> [Linear: 64 -> 1] -> Output: ln(SalePrice)",
        bold_prefix="Cấu trúc Mạng MLPRegressor:"
    )

    add_heading_3("A. Quá trình Lan truyền tiến (Forward Propagation)")
    doc.add_paragraph("Đối với mỗi tầng ẩn l in {1, 2, ..., L}:")
    add_formula("z^[l] = W^[l] * a^[l-1] + b^[l]")
    doc.add_paragraph("Hàm kích hoạt phi tuyến Rectified Linear Unit (ReLU):")
    add_formula("a^[l] = max(0, z^[l])")
    doc.add_paragraph(
        "Ưu điểm của ReLU so với Sigmoid/Tanh: Đạo hàm của ReLU bằng hằng số 1 với mọi z > 0, triệt tiêu hoàn toàn hiện tượng triệt tiêu đạo hàm (Vanishing Gradient Problem) khi truyền ngược qua các tầng sâu, đồng thời mang lại tính thưa (sparsity) cho mạng nơ-ron giúp tăng tốc tính toán."
    )
    doc.add_paragraph("Tầng đầu ra gồm 1 nơ-ron tuyến tính thuần túy để dự báo giá trị liên tục:")
    add_formula("y_hat_log = w^[L]T * a^[L-1] + b^[L]")

    add_heading_3("B. Hàm mất mát (Loss Function)")
    doc.add_paragraph("Hàm mất mát Mean Squared Error (MSE) được tính trên mini-batch đối với biến mục tiêu log:")
    add_formula("L_MSE = (1 / B) * sum_{i=1}^{B} ( y_hat_log_i - y_log_i )^2")

    add_heading_3("C. Thuật toán Tối ưu hóa Adam (Adaptive Moment Estimation)")
    doc.add_paragraph(
        "Mạng được tối ưu bằng thuật toán Adam kết hợp giữa phương pháp quán tính (Momentum) và chia tỷ lệ gradient theo tọa độ (RMSProp):"
    )
    add_formula("m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t   (Momentum bậc 1)")
    add_formula("v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2 (RMSProp bậc 2)")
    add_formula("theta_{t+1} = theta_t - ( eta / (sqrt(v_hat_t) + eps) ) * m_hat_t")
    doc.add_paragraph("Siêu tham số thực nghiệm: tốc độ học eta = 0.001, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8, batch_size = 32.")

    add_heading_3("D. Các kỹ thuật Điều hòa (Regularization) chống Quá khớp")
    add_bullet("Dropout (p = 0.2): Ngắt ngẫu nhiên 20% số nơ-ron ở mỗi lượt truyền tiến để ngăn chặn hiện tượng đồng thích nghi (co-adaptation) giữa các nơ-ron.")
    add_bullet("Weight Decay (L2 = 1e-4): Bổ sung số hạng phạt trọng số lớn vào hàm mục tiêu.")
    add_bullet("Early Stopping (Patience = 20): Dừng huấn luyện khi validation loss không cải thiện liên tiếp 20 epoch và lưu lại checkpoint tốt nhất (mlp_model_best.pth).")

    add_heading_2("4.2. Xây dựng Mô hình Đối chuẩn (Baselines)")
    add_bullet("Ordinary Least Squares (OLS) Linear Regression: Hồi quy tuyến tính cổ điển tìm nghiệm giải tích tối ưu không có thành phần phạt.")
    add_bullet("Ridge Regression (alpha = 1.0): Hồi quy tuyến tính có chính quy hóa chuẩn L2 nhằm kiểm soát hiện tượng đa cộng tuyến trên không gian 318 chiều.")
    add_bullet("Random Forest Regressor: Mô hình học kết hợp (bagging ensemble) gồm 100 cây quyết định để chụp các mối tương tác phi tuyến tính.")

    # --- PHASE 5: EVALUATION ---
    add_heading_1("GIAI ĐOẠN 5: EVALUATION (ĐÁNH GIÁ & SO SÁNH CHUYÊN SÂU)")
    add_heading_2("5.1. Bảng Tổng hợp Kết quả Đánh giá Thực nghiệm")
    doc.add_paragraph(
        "Toàn bộ các mô hình được đánh giá trên cùng tập kiểm định Validation (292 mẫu, 20% dữ liệu train, seed 42). Các chỉ số đánh giá gồm: log-RMSE, log-MAE và hệ số xác định R2:"
    )

    eval_table = doc.add_table(rows=5, cols=5)
    eval_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    ev_headers = ["STT", "Mô hình (Model)", "Validation log-RMSE", "Validation log-MAE", "Validation R²"]
    ev_widths = [Inches(0.6), Inches(2.3), Inches(1.3), Inches(1.3), Inches(1.1)]

    ev_hdr_row = eval_table.rows[0]
    for j, h in enumerate(ev_headers):
        cell = ev_hdr_row.cells[j]
        cell.width = ev_widths[j]
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=100, bottom=100, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(10.5)

    ev_data = [
        ("1", "Linear Regression (OLS)", "0.1332", "0.0955", "0.9049"),
        ("2", "Ridge Regression (alpha=1.0)", "0.1363", "0.0973", "0.9004"),
        ("3", "PyTorch MLP (Deep Learning)", "0.1447", "0.1005", "0.8878"),
        ("4", "Random Forest Baseline", "0.1517", "0.0992", "0.8767")
    ]

    set_table_borders(eval_table, color="D3D3D3", sz="4")
    for i, data in enumerate(ev_data):
        row = eval_table.rows[i+1]
        bg = "F9FBFD" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(data):
            cell = row.cells[j]
            cell.width = ev_widths[j]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            if j in [0, 2, 3, 4]:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(10)
            if i == 0 or (i == 2 and j == 1):
                r.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    add_image_safe(str(figures_dir / "actual_vs_predicted_mlp.png"), "Hình 5: Đồ thị tương quan giữa Giá trị thực tế và Giá trị dự đoán của mô hình PyTorch MLP trên tập Validation.")

    add_heading_2("5.2. Luận cứ Khoa học: Vì sao Linear/Ridge lại vượt trội hơn Mạng Nơ-ron MLP trên tập dữ liệu này?")
    doc.add_paragraph(
        "Một kết quả thực nghiệm nổi bật và mang tính học thuật cao trong Lab 02 là: Mô hình hồi quy tuyến tính cổ điển (Linear Regression) và Ridge Regression lại cho kết quả vượt trội hơn so với mạng nơ-ron học sâu Multilayer Perceptron (MLP) (log-RMSE 0.1332 và 0.1363 so với 0.1447). Hiện tượng này hoàn toàn nhất quán với các công trình nghiên cứu hiện đại về học sâu cho dữ liệu dạng bảng (Tabular Data) như Kadra et al. (NeurIPS 2021) hay Grinsztajn et al. (NeurIPS 2022). Nguyên nhân khoa học được lý giải qua 4 luận điểm then chốt:"
    )

    add_heading_3("Luận điểm 1: Lời nguyền số chiều và Sự khan hiếm dữ liệu (Curse of Dimensionality & Small Sample Size)")
    doc.add_paragraph(
        "Tập dữ liệu huấn luyện sau khi chia chỉ có 1.168 mẫu, nhưng không gian đặc trưng sau khi One-Hot Encoding đã phình to lên đến 318 chiều. Xét về số lượng tham số tự do (parameters):"
    )
    add_bullet("Ridge / Linear Regression: Chỉ có đúng 318 trọng số + 1 hệ số bias = 319 tham số cần tối ưu. Số bậc tự do hoàn toàn tương thích với quy mô 1.168 mẫu (N > D).")
    add_bullet("PyTorch MLP (Kiến trúc 318 - 128 - 64 - 1): Tổng cộng có 49.153 tham số tự do (Tầng 1: 40.832, Tầng 2: 8.256, Tầng 3: 65).")
    doc.add_paragraph(
        "Tỷ lệ dữ liệu trên tham số của MLP là 1.168 / 49.153 ~ 0.0238 (trung bình chỉ có khoảng 0.02 điểm dữ liệu để ước lượng một tham số nơ-ron!). Với năng lực biểu diễn quá lớn (high capacity) trên một tập dữ liệu quá nhỏ, dù có áp dụng Dropout và Early Stopping, mạng nơ-ron vẫn có xu hướng ghi nhớ nhiễu cục bộ của tập huấn luyện hơn là học được quy luật tổng quát."
    )

    add_heading_3("Luận điểm 2: Thiên kiến quy nạp (Inductive Bias) và Bản chất cộng dồn của Giá bất động sản")
    doc.add_paragraph(
        "Trong kinh tế học lượng giác và định giá tài sản, giá trị của một ngôi nhà có bản chất cộng dồn tuyến tính (Additive Nature) rất mạnh: Giá nhà ~ (Đơn giá đất x Diện tích) + (Chất lượng x Hệ số vật liệu) + Giá trị tiện ích thêm. Mô hình tuyến tính sở hữu thiên kiến quy nạp (Inductive Bias) hoàn hảo cho giả thiết này vì nó giả định mối quan hệ giữa biến đầu vào và đầu ra là siêu phẳng tuyến tính."
    )
    doc.add_paragraph(
        "Ngược lại, mạng MLP không có bất kỳ giả thiết tiên nghiệm nào về cấu trúc cộng dồn này. MLP phải tự mình xấp xỉ lại các phép cộng đơn giản thông qua sự kết hợp phức tạp của hàng chục nghìn nơ-ron và hàm kích hoạt phi tuyến ReLU. Điều này gây lãng phí năng lực tính toán và dễ tạo ra các sai số xấp xỉ không đáng có."
    )

    add_heading_3("Luận điểm 3: Bản chất toán học của Không gian Tối ưu hóa (Optimization Landscape)")
    doc.add_paragraph(
        "Đối với Ridge Regression: Hàm mục tiêu là một hàm lồi ngặt (Strictly Convex Optimization Problem). Bài toán luôn tồn tại một nghiệm cực tiểu toàn cục duy nhất dạng đóng (Closed-form Analytical Solution) thông qua phép nghịch đảo ma trận (X^T * X + alpha * I)^(-1). Quá trình tìm nghiệm không phụ thuộc vào giá trị khởi tạo và không bao giờ bị kẹt ở các điểm cực tiểu cục bộ."
    )
    doc.add_paragraph(
        "Đối với Mạng Nơ-ron MLP: Bề mặt hàm mất mát là phi lồi (Non-convex), chứa vô số điểm yên ngựa (saddle points) và cực tiểu địa phương (local minima). Các thuật toán tối ưu hóa ngẫu nhiên (như Adam/SGD) cùng với kích thước batch nhỏ (B=32) tạo ra các bước nhảy gradient ngẫu nhiên lớn, khiến mạng khó có thể hội tụ chính xác đến điểm tối ưu tuyệt đối trên không gian mẫu hạn hẹp."
    )

    add_heading_3("Luận điểm 4: Đặc thù của Dữ liệu dạng bảng (Tabular Data Heterogeneity) so với Cây quyết định")
    doc.add_paragraph(
        "Dữ liệu dạng bảng không có cấu trúc không gian (như ảnh - CNN) hay cấu trúc chuỗi thời gian (như văn bản - Transformer). Các cột dữ liệu hoàn toàn không đồng nhất: có cột liên tục thang đo lớn, có cột nhị phân thưa (0/1 sau OHE)."
    )
    doc.add_paragraph(
        "Mô hình Random Forest tuy mạnh nhưng trong bài toán này lại đạt kết quả kém nhất (log-RMSE = 0.1517). Lý do là các cây quyết định phân chia không gian bằng các siêu phẳng trực giao với các trục tọa độ (axis-aligned splits), tạo ra các hàm dự đoán dạng bậc thang (step functions). Cấu trúc bậc thang này rất khó xấp xỉ mượt mà một hàm hồi quy tuyến tính liên tục như giá nhà, đồng thời cây quyết định bị suy giảm hiệu năng nghiêm trọng khi đối mặt với không gian thưa 318 chiều."
    )

    # --- PHASE 6: DEPLOYMENT ---
    add_heading_1("GIAI ĐOẠN 6: DEPLOYMENT (TRIỂN KHAI & TÁI LẬP KẾT QUẢ)")
    add_heading_2("6.1. Kiến trúc Đóng gói Mã nguồn Chuyên nghiệp (src/)")
    doc.add_paragraph(
        "Toàn bộ dự án đã được tái cấu trúc từ các notebook nghiên cứu ban đầu thành một gói thư viện Python hoàn chỉnh, chuẩn mực theo kiến trúc MLOps:"
    )
    add_bullet("src/config.py: Tự động định vị đường dẫn thư mục gốc LAB02_ROOT bằng pathlib.Path, quản lý tập trung toàn bộ siêu tham số thực nghiệm (learning rate, batch size, epochs, random seed) linh hoạt trên mọi hệ điều hành.")
    add_bullet("src/data/dataset.py: Định nghĩa lớp HousePriceDataset chuẩn PyTorch kế thừa từ torch.utils.data.Dataset, hỗ trợ nhận dữ liệu từ DataFrame hoặc Tensor, chuyển đổi dữ liệu thành các mini-batch phục vụ DataLoader.")
    add_bullet("src/models/mlp.py: Khai báo lớp mô hình MLPRegressor kế thừa nn.Module, cho phép cấu hình linh hoạt số tầng ẩn, số nơ-ron và xác suất dropout.")
    add_bullet("src/utils/metrics.py: Module tính toán độc lập các chỉ số đánh giá (RMSE, MAE, R2) trên cả hai thang đo (log-scale và thang đo giá gốc đô la USD).")

    add_heading_2("6.2. Quy trình Dự báo và Chuẩn bị Nộp bài Kaggle")
    doc.add_paragraph(
        "Toàn bộ pipeline tiền xử lý và trọng số mô hình tốt nhất (models/mlp_model_best.pth) được nạp để thực hiện suy diễn (Inference) trên tập kiểm thử ẩn data/raw/test.csv (1.459 dòng). Do mô hình được huấn luyện trên biến mục tiêu biến đổi ln(1 + SalePrice), đầu ra dự đoán của mạng y_hat_log được khôi phục về thang đo giá trị tiền tệ thực tế thông qua hàm số mũ:"
    )
    add_formula("y_hat_dollar = exp(y_hat_log) - 1")
    doc.add_paragraph(
        "Tệp kết quả được xuất ra tại outputs/submissions/submission_mlp.csv với đúng 1.459 dòng dự đoán và 2 cột chuẩn xác theo yêu cầu cuộc thi: Id và SalePrice. Kiểm tra tính hợp lệ xác nhận tệp không chứa giá trị NaN, không có giá trị âm hoặc vô cùng, định dạng hoàn toàn khớp với sample_submission.csv và sẵn sàng ghi nhận điểm xếp hạng công khai (Public Leaderboard Score)."
    )
    add_image_safe(str(figures_dir / "submission.png"), "Hình 6: Kiểm tra tính hợp lệ và cấu trúc định dạng của tệp nộp bài outputs/submissions/submission_mlp.csv.")

    add_heading_2("6.3. Khả năng Tái lập (Reproducibility) và Hướng phát triển trong tương lai")
    add_bullet("Khả năng tái lập: Toàn bộ quá trình từ nạp dữ liệu thô, tiền xử lý, huấn luyện baseline đến huấn luyện PyTorch MLP được cố định bằng một seed duy nhất (SEED = 42). Bất kỳ nhà nghiên cứu nào khi tải kho mã nguồn đều có thể chạy lại từ đầu và tái lập 100% các con số sai số trong báo cáo này.")
    add_bullet("Định hướng Feature Engineering chuyên sâu: Tích hợp các biến kết hợp mang tri thức nghiệp vụ sâu như: Tổng diện tích sinh hoạt thực tế (TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF), Tuổi thọ căn nhà lúc bán (Age = YrSold - YearBuilt), Tổng số phòng tắm (TotalBath = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath).")
    add_bullet("Định hướng Kiến trúc Deep Learning tiên tiến: Thử nghiệm mô hình TabNet (Arik & Pfister, Google Cloud AI) sử dụng cơ chế Sequential Attention để chọn lọc đặc trưng động hoặc FT-Transformer (Gorishniy et al., 2021).")
    add_bullet("Định hướng Ensembling: Kết hợp sức mạnh của mô hình tuyến tính chính quy hóa (Ridge/Lasso) cùng với Gradient Boosting (XGBoost, LightGBM, CatBoost) và PyTorch MLP thông qua kỹ thuật Stacking hoặc Blending để đạt thứ hạng tối ưu trên bảng xếp hạng Kaggle toàn cầu.")

    # --- REFERENCES ---
    add_heading_1("TÀI LIỆU THAM KHẢO (REFERENCES)")
    refs = [
        "De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education, 19(3), 1–14.",
        "Kaggle Inc. House Prices: Advanced Regression Techniques. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques.",
        "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems (NeurIPS 2019), 32.",
        "Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv preprint arXiv:1412.6980.",
        "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. The Journal of Machine Learning Research, 15(1), 1929–1958.",
        "Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data?. Advances in Neural Information Processing Systems (NeurIPS 2022), 35, 507–520.",
        "Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., & Wirth, R. (2000). CRISP-DM 1.0: Step-by-step data mining guide. SPSS inc, 9–13."
    ]
    for r in refs:
        add_bullet(r)

    # Save Document
    out_docx_path = root_dir / "docs" / "reports" / "Lab02_House_Prices_Prediction_CRISP_DM_Report.docx"
    doc.save(str(out_docx_path))
    print(f"Báo cáo Word đã được tạo thành công tại: {out_docx_path}")
    print(f"Kích thước file: {os.path.getsize(out_docx_path)} bytes")

if __name__ == "__main__":
    build_docx_report()
