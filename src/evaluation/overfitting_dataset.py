# Dataset untuk evaluasi generalisasi / overfitting sistem RAG
# Dataset A  : pertanyaan yang sama persis dengan dataset evaluasi utama (referensi)
# Dataset B  : pertanyaan baru — topik sama tapi sudut pandang, kata, dan
#              struktur kalimat BERBEDA (bukan sekadar parafrase kata demi kata)
# Setiap entri memiliki ground_truth sehingga kualitas jawaban bisa diukur.

# ── Dataset A — pertanyaan referensi (dari TEST_DATASET) ─────────────────────
DATASET_A = [
    {
        "kategori"   : "definisi",
        "question"   : "Apa yang dimaksud dengan Purchase Order?",
        "ground_truth": (
            "Purchase Order (PO) adalah dokumen yang digunakan untuk order "
            "barang dan dikirimkan kepada vendor sebagai tanda jadi atas order "
            "yang dilakukan. Purchase Order berisi informasi spesifikasi dan "
            "harga yang telah disepakati dan disetujui oleh atasan."
        )
    },
    {
        "kategori"   : "dokumen",
        "question"   : "Apa saja dokumen yang diperlukan untuk pengajuan kasbon dosen tamu?",
        "ground_truth": (
            "Dokumen yang diperlukan untuk pengajuan kasbon dosen tamu adalah: "
            "RF (Requisition Form), Form Rekap Fee, NPWP/KTP."
        )
    },
    {
        "kategori"   : "definisi",
        "question"   : "Apa yang dimaksud dengan Requisition Form?",
        "ground_truth": (
            "Requisition Form (RF) adalah formulir untuk internal perusahaan "
            "yang berfungsi untuk mencatat permintaan pembelian barang kepada "
            "bagian pembelian."
        )
    },
    {
        "kategori"   : "numerik",
        "question"   : "Berapa besaran retensi minimal dalam pembayaran SPK?",
        "ground_truth": (
            "Besaran retensi minimal adalah 5% dan lama retensi minimal "
            "adalah 3 bulan."
        )
    },
    {
        "kategori"   : "definisi",
        "question"   : "Apa yang dimaksud dengan MBKM?",
        "ground_truth": (
            "MBKM adalah Merdeka Belajar Kampus Merdeka, yaitu kebijakan "
            "dari Menteri Pendidikan dan Kebudayaan yang memberikan kebebasan "
            "bagi mahasiswa untuk belajar di luar program studi."
        )
    },
    {
        "kategori"   : "prosedur",
        "question"   : "Bagaimana proses transfer SKS MBKM?",
        "ground_truth": (
            "Proses transfer SKS MBKM melibatkan: tim asesor melakukan "
            "pemeriksaan beban mata kuliah, menetapkan kesetaraan mata kuliah, "
            "menetapkan pengakuan SKS, mengusulkan transfer ke BAA, BAA "
            "melakukan transfer SKS dan nilai, serta Programmer section head "
            "melakukan pelaporan ke PDDikti."
        )
    },
    {
        "kategori"   : "penanggung jawab",
        "question"   : "Siapa yang bertanggung jawab melakukan pelaporan kegiatan MBKM ke PDDikti?",
        "ground_truth": (
            "Programmer Section Head bertanggung jawab melakukan pelaporan "
            "kegiatan MBKM pada laman PDDikti sesuai dengan periode yang "
            "telah ditetapkan PDDikti."
        )
    },
    {
        "kategori"   : "numerik",
        "question"   : "Berapa batas waktu pembayaran setelah berkas tagihan diterima?",
        "ground_truth": (
            "Pembayaran dilakukan maksimal 2 minggu setelah berkas tagihan "
            "diterima."
        )
    },
]

# ── Dataset B — pertanyaan generalisasi (sudut pandang & konteks berbeda) ─────
# Kriteria Dataset B:
# - Topik sama dengan Dataset A, tapi pertanyaan datang dari SUDUT PANDANG BARU
# - Bukan sekadar mengganti kata (parafrase), tapi menanyakan aspek berbeda
# - Memiliki ground_truth yang jelas dari dokumen
DATASET_B = [
    {
        "kategori"   : "perbedaan konsep",
        "question"   : "Apa perbedaan antara Requisition Form dan Purchase Order dalam proses pengadaan?",
        "ground_truth": (
            "Requisition Form (RF) adalah formulir internal untuk mencatat "
            "permintaan pembelian barang kepada bagian pembelian. Sedangkan "
            "Purchase Order (PO) adalah dokumen yang dikirimkan kepada vendor "
            "sebagai tanda jadi atas order yang dilakukan, berisi spesifikasi "
            "dan harga yang telah disepakati dan disetujui oleh atasan."
        )
    },
    {
        "kategori"   : "dokumen",
        "question"   : "Dokumen apa saja yang harus dilampirkan saat mengajukan reimburse untuk dosen tamu?",
        "ground_truth": (
            "Dokumen yang diperlukan untuk pengajuan reimburse dosen tamu adalah: "
            "RF (Requisition Form), Laporan pertanggungjawaban keuangan, "
            "NPWP/KTP, Attendance List, Kuitansi."
        )
    },
    {
        "kategori"   : "tujuan program",
        "question"   : "Apa tujuan dari program Merdeka Belajar Kampus Merdeka bagi mahasiswa?",
        "ground_truth": (
            "Tujuan MBKM adalah memberikan kebebasan bagi mahasiswa untuk "
            "belajar di luar program studi dan/atau di luar perguruan tinggi "
            "yang sedang dijalani, berdasarkan kebijakan Menteri Pendidikan "
            "dan Kebudayaan."
        )
    },
    {
        "kategori"   : "peran unit",
        "question"   : "Apa peran BAA dalam proses transfer SKS mahasiswa peserta MBKM?",
        "ground_truth": (
            "BAA melakukan transfer SKS dan nilai sesuai dengan permohonan "
            "yang diajukan oleh tim asesor dan program studi, apabila dokumen "
            "yang diserahkan sudah sesuai."
        )
    },
    {
        "kategori"   : "ketentuan kontrak",
        "question"   : "Berapa lama dan berapa persen retensi yang harus ditahan dalam kontrak SPK?",
        "ground_truth": (
            "Retensi minimal yang ditahan adalah 5% dari nilai kontrak, "
            "dan lama retensi minimal adalah 3 bulan."
        )
    },
    {
        "kategori"   : "peran unit",
        "question"   : "Bagian mana yang bertanggung jawab melaporkan program MBKM ke sistem PDDikti?",
        "ground_truth": (
            "Programmer section head bertanggung jawab melakukan pelaporan "
            "pada sistem pendokumentasian program MBKM dan pada laman PDDikti "
            "sesuai dengan periode yang telah ditetapkan."
        )
    },
    {
        "kategori"   : "prosedur",
        "question"   : "Apa yang dilakukan tim asesor saat memproses pengakuan SKS mahasiswa yang kuliah di PT lain?",
        "ground_truth": (
            "Tim asesor melakukan pemeriksaan beban mata kuliah yang diambil "
            "mahasiswa di PT lain, menetapkan kesetaraan atau ketidaksetaraan "
            "mata kuliah dengan kurikulum prodi asal, dan menetapkan pengakuan "
            "jumlah SKS yang dapat disetarakan."
        )
    },
    {
        "kategori"   : "numerik",
        "question"   : "Berapa jatuh tempo pembayaran invoice tagihan biaya kuliah mahasiswa?",
        "ground_truth": (
            "Jatuh tempo pembayaran invoice tagihan biaya kuliah mahasiswa "
            "adalah 3 hari setelah pembuatan invoice, dengan batas acc "
            "invoice oleh Vice Head FA."
        )
    },
]
