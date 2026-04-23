# Dataset pengujian untuk evaluasi RAGAS
# Format: question, ground_truth (jawaban benar dari dokumen)

TEST_DATASET = [
    {
        "question": "Apa yang dimaksud dengan Purchase Order?",
        "ground_truth": (
            "Purchase Order (PO) adalah dokumen yang digunakan untuk "
            "order barang dan dikirimkan kepada vendor sebagai tanda jadi "
            "atas order yang dilakukan. Purchase Order berisi informasi "
            "spesifikasi dan harga yang telah disepakati dan disetujui "
            "oleh atasan."
        )
    },
    {
        "question": "Apa saja dokumen yang diperlukan untuk pengajuan kasbon dosen tamu?",
        "ground_truth": (
            "Dokumen yang diperlukan untuk pengajuan kasbon dosen tamu adalah: "
            "RF (Requisition Form), Form Rekap Fee, NPWP/KTP."
        )
    },
    {
        "question": "Apa saja dokumen yang diperlukan untuk pengajuan reimburse dosen tamu?",
        "ground_truth": (
            "Dokumen yang diperlukan untuk pengajuan reimburse dosen tamu adalah: "
            "RF (Requisition Form), Laporan pertanggungjawaban keuangan, "
            "NPWP/KTP, Attendance List, Kuitansi."
        )
    },
    {
        "question": "Berapa lama masa proses pengeluaran dana untuk nominal di atas Rp 100.000.000?",
        "ground_truth": (
            "Masa proses pengeluaran dana untuk nominal di atas "
            "Rp 100.000.000 adalah maksimal 14 hari kerja."
        )
    },
    {
        "question": "Apa yang dimaksud dengan MBKM?",
        "ground_truth": (
            "MBKM adalah Merdeka Belajar Kampus Merdeka, yaitu kebijakan "
            "dari Menteri Pendidikan dan Kebudayaan yang memberikan "
            "kebebasan bagi mahasiswa untuk belajar di luar program studi."
        )
    },
    {
        "question": "Siapa yang bertanggung jawab melakukan pelaporan kegiatan MBKM ke PDDikti?",
        "ground_truth": (
            "Programmer Section Head bertanggung jawab melakukan pelaporan "
            "kegiatan MBKM pada laman PDDikti sesuai dengan periode yang "
            "telah ditetapkan PDDikti."
        )
    },
    {
        "question": "Bagaimana proses transfer SKS bagi mahasiswa STIE Ciputra Makassar yang mengambil mata kuliah di PT lain?",
        "ground_truth": (
            "Proses transfer SKS adalah: "
            "1. Tim asesor melakukan pemeriksaan beban mata kuliah yang diambil. "
            "2. Tim asesor dan prodi menetapkan kesetaraan mata kuliah. "
            "3. Tim asesor dan prodi menetapkan pengakuan jumlah SKS. "
            "4. Tim asesor dan prodi mengusulkan transfer SKS ke BAA "
            "menggunakan form UCM/FR/BAA/031 alih kredit. "
            "5. BAA melakukan transfer SKS sesuai permohonan. "
            "6. Programmer section head melakukan pelaporan ke PDDikti."
        )
    },
    {
        "question": "Berapa besaran retensi minimal dalam pembayaran SPK?",
        "ground_truth": (
            "Besaran retensi minimal adalah 5% dan lama retensi minimal "
            "adalah 3 bulan."
        )
    },
    {
        "question": "Berapa batas waktu pembayaran setelah berkas tagihan diterima?",
        "ground_truth": (
            "Pembayaran dilakukan maksimal 2 minggu setelah berkas "
            "tagihan diterima."
        )
    },
    {
        "question": "Apa yang dimaksud dengan Requisition Form?",
        "ground_truth": (
            "Requisition Form (RF) adalah formulir untuk internal "
            "perusahaan yang berfungsi untuk mencatat permintaan "
            "pembelian barang kepada bagian pembelian."
        )
    },
]