# chat.py
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

class_info_dict = {
    # Daun Sehat
    "Apple leaf": "Ini merupakan daun apel yang sehat. Pohon apel adalah tanaman gugur yang banyak dibudidayakan untuk buahnya yang lezat.",
    "Healthy Chili Leaf": "Ini merupakan daun cabai yang sehat. Cabai adalah varietas Capsicum annuum yang dibudidayakan untuk buahnya yang digunakan dalam masakan.",
    "Blueberry leaf": "Ini merupakan daun blueberry yang sehat. Blueberry adalah semak kecil yang menghasilkan buah kaya antioksidan.",
    "Cherry leaf": "Ini merupakan daun ceri yang sehat. Pohon ceri menghasilkan buah kecil berbentuk bulat yang biasanya berwarna merah atau hitam.",
    "Corn leaf": "Ini merupakan daun jagung yang sehat. Jagung adalah tanaman serealia penting yang digunakan untuk konsumsi manusia maupun pakan ternak.",
    "Peach leaf": "Ini merupakan daun persik yang sehat. Pohon persik menghasilkan buah manis dan berair yang dibudidayakan di iklim sedang.",
    "Potato leaf": "Ini merupakan daun kentang yang sehat. Kentang adalah tanaman umbi yang menjadi makanan pokok di banyak negara.",
    "Raspberry leaf": "Ini merupakan daun raspberry yang sehat. Tanaman raspberry menghasilkan buah kecil berwarna merah atau hitam yang kaya nutrisi.",
    "Soybean leaf": "Ini merupakan daun kedelai yang sehat. Kedelai adalah tanaman legum dan sumber protein serta minyak yang penting.",
    "Strawberry leaf": "Ini merupakan daun stroberi yang sehat. Tanaman stroberi dikenal dengan buahnya yang manis dan berwarna merah.",
    "Tomato leaf": "Ini merupakan daun tomat yang sehat. Tanaman tomat menghasilkan buah merah atau kuning yang merupakan bahan penting dalam banyak masakan.",
    "Grape leaf": "Ini merupakan daun anggur yang sehat. Anggur dibudidayakan untuk dikonsumsi langsung maupun untuk produksi anggur.",

    # Daun Penyakit
    "Apple Scab Leaf": "Apple scab adalah penyakit yang disebabkan oleh jamur Venturia inaequalis. Penyakit ini menghasilkan lesi gelap seperti kerak pada daun.",
    "Apple rust leaf": "Apple rust disebabkan oleh berbagai jenis jamur dan menghasilkan bercak berwarna kuning-oranye pada daun.",
    "Chili Leaf Spot": "Ini adalah penyakit jamur atau bakteri yang menyebabkan munculnya bercak kecil gelap pada daun cabai.",
    "Corn Gray leaf spot": "Disebabkan oleh jamur Cercospora zeae-maydis, penyakit ini menghasilkan bercak abu-abu berbentuk persegi panjang pada daun jagung.",
    "Corn leaf blight": "Disebabkan oleh jamur Helminthosporium maydis, penyakit ini menyebabkan garis-garis abu-abu panjang pada daun jagung.",
    "Corn rust leaf": "Penyakit ini disebabkan oleh jamur Puccinia sorghi dan menghasilkan bercak yang mirip karat pada daun jagung.",
    "Potato leaf early blight": "Disebabkan oleh jamur Alternaria solani, penyakit ini menghasilkan bercak gelap pada daun kentang.",
    "Potato leaf late blight": "Disebabkan oleh oomycete Phytophthora infestans, penyakit ini dapat menyebabkan lesi besar berwarna gelap pada daun kentang.",
    "Squash Powdery mildew leaf": "Penyakit ini ditandai dengan munculnya bercak putih seperti bedak dan disebabkan oleh berbagai spesies jamur.",
    "Tomato Early blight leaf": "Disebabkan oleh jamur Alternaria solani, penyakit ini menghasilkan bercak gelap pada daun tomat.",
    "Tomato Septoria leaf spot": "Disebabkan oleh jamur Septoria lycopersici, penyakit ini menghasilkan bercak kecil gelap dengan pusat yang lebih terang pada daun tomat.",
    "Tomato leaf bacterial spot": "Penyakit ini disebabkan oleh bakteri dan menghasilkan bercak nekrotik kecil pada daun tomat.",
    "Tomato leaf late blight": "Serupa dengan late blight pada kentang, penyakit ini disebabkan oleh Phytophthora infestans pada tomat.",
    "Tomato leaf mosaic virus": "Penyakit virus ini menyebabkan daun menjadi cacat dan berbintik-bintik.",
    "Tomato leaf yellow virus": "Penyakit virus yang menyebabkan daun menguning dan melengkung.",
    "Tomato mold leaf": "Penyakit ini bisa disebabkan oleh berbagai jenis jamur mold yang menghasilkan pertumbuhan berbulu pada daun tomat.",
    "Tomato two spotted spider mites leaf": "Ini merupakan infestasi kutu laba-laba dua bintik, bukan penyakit, yang menyebabkan bintik dan perubahan warna pada daun.",
    "Grape leaf black rot": "Disebabkan oleh jamur Guignardia bidwellii, penyakit ini menghasilkan bercak hitam pada daun anggur.",
    "Whitefly Chili Leaf": "Serangan Whitefly pada daun cabai dapat menyebabkan daun menguning dan tanaman menjadi lemah akibat serangannya.",
    "Yellow Chili Leaf": "Daun cabai yang menguning dapat disebabkan oleh kekurangan nutrisi, serangan hama, atau penyakit.",
    "Curly Chili Leaf": "Daun cabai yang mengeriting bisa menjadi tanda serangan hama thrips atau infeksi virus kuning.",
}

def chatbot(message):
    prompt_template = """
    You are a helpful farming assistant specialized in plant health and disease detection.

    Instructions:
    - If the user asks about plants or plant diseases, answer with short and clear agricultural advice.
    - If the user greets you (e.g., hello, hi), respond with a friendly greeting.
    - If the user says goodbye, respond with a polite farewell.
    - If the user says thank you or expresses gratitude, respond warmly and say you're happy to help.
    - If the user asks something unrelated to plants (e.g., politics, weather, sports, tech, etc.), politely respond with:
      "Maaf, saya hanya bisa membantu pertanyaan yang berkaitan dengan tanaman dan penyakit daun."

    User message: {message}
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['message']
    )

    chain = LLMChain(llm=llm, prompt=PROMPT)
    response = chain.predict(message=message)

    return response


def generate_deskripsi(info):
    prompt_template = """
    Anda adalah seorang ahli patologi tanaman. Berdasarkan informasi penyakit yang terdeteksi berikut:
    {info}

    Berikan insight yang komprehensif dan langsung menjelaskan tanpa 'terima kasih' terkait Deskripsi Penyakitnya seperti Jelaskan apa itu penyakit tersebut, gejalanya, dan dampaknya terhadap tanaman

    Silakan sampaikan jawaban Anda dengan jelas, ringkas maksimal 1500 karakter dan mudah dipahamin orang awam.
    """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['info']
    )

    chain = LLMChain(llm=llm, prompt=PROMPT)
    response = chain.predict(info=info)
    return response

def generate_pencegahan(info):
    prompt_template = """
    Anda adalah seorang ahli patologi tanaman. Berdasarkan informasi penyakit yang terdeteksi berikut:
    {info}

    Berikan insight yang komprehensif dan langsung menjelaskan tanpa 'terima kasih' terkait pencegahan Penyakitnya seperti Jelaskan metode efektif untuk mencegah munculnya atau penyebaran penyakit ini dalam bentuk LIST.
    Jelaskan metode efektif untuk mencegah munculnya atau penyebaran penyakit ini dengan menyebutkan maksimal 3 point terpenting tanpa memberikan tanda simbol bintang / **.


    Silakan sampaikan jawaban Anda dengan jelas, ringkas dan mudah dipahamin orang awam.
    """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['info']
    )

    chain = LLMChain(llm=llm, prompt=PROMPT)
    response = chain.predict(info=info)
    return response

def generate_penanganan(info):
    prompt_template = """
    Anda adalah seorang ahli patologi tanaman. Berdasarkan informasi penyakit yang terdeteksi berikut:
    {info}

    Berikan insight yang komprehensif dan langsung menjelaskan tanpa 'terima kasih' terkait Penanganan Penyakitnya.
    Berikan rekomendasi opsi pengobatan praktis untuk mengelola atau menyembuhkan penyakit ini dalam bentuk LIST.
    Selain itu, tentukan dan cantumkan parameter usia tanaman yang relevan untuk penanganan, misalnya dengan:
      - Menentukan apakah tanaman berada pada fase awal, pertengahan, atau fase akhir pertumbuhan.
      - Memberikan opsi penanganan yang sesuai berdasarkan rentang usia yang Anda anggap optimal.
    Pastikan untuk menyebutkan maksimal 3 point terpenting sebagai rekomendasi penanganan tanpa memberikan tanda simbol bintang / **..
    Silakan sampaikan jawaban Anda dengan jelas, ringkas, dan mudah dipahami oleh orang awam.
    """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['info']
    )
    chain = LLMChain(llm=llm, prompt=PROMPT)
    response = chain.predict(info=info)
    return response

