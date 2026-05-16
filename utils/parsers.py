# 文件解析工具类：统一解析 TXT / PDF / DOCX 文件
import os
from pypdf import PdfReader
from docx import Document
from utils.logger import logger

class FileParser:
    """
    文件解析器：根据文件后缀名，自动选择对应解析方式
    支持：.txt / .pdf / .docx
    """

    @staticmethod   #定义静态方法：不需要创建对象直接调用
    def parse(file_path: str) -> str:
        """
        外部统一调用这个函数
        输入：文件路径
        输出：文件里的全部文本内容
        """
        
        #获取文件后缀（转小写）
        ext = os.path.splitext(file_path)[-1].lower()

        # 根据后缀，自动选解析器
        if ext == ".txt" or ext == ".md":
            return FileParser._parse_txt(file_path)
        
        elif ext == ".pdf":
            return FileParser._parse_pdf(file_path)

        elif ext == ".docx":
            return FileParser._parse_docx(file_path)

        else:
            logger.warning(f"暂时还不支持'{ext}'的文件类型") 
            return ""
    
    @staticmethod
    def _parse_txt(file_path: str) -> str:
        """解析 .txt 文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        except Exception as e:
            logger.error(f"TXT解析失败{e}")

    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        """解析 .pdf 文件"""
        try:
            reader = PdfReader(file_path)
            text = ""
            # 一页一页读
            for page in reader.pages:
                page_text = page.extract_text() #提取当前页的所有文字
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"PDF解析失败：{e}")
            return ""
        
    @staticmethod
    def _parse_docx(file_path: str) -> str:
        """解析 .docx 文件"""
        try:
            doc = Document(file_path)   #打开文档
            
            # 读所有段落并以列表格式存储，在使用换行符拼接每一段
            text = "\n".join([para.text for para in doc.paragraphs])

            return text
        except Exception as e:
            logger.error(f"DOCX文件解析失败：{e}")
            return ""

            