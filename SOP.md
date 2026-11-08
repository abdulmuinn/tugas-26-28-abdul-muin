# SOP — Penanganan Alert Model Kredit `credit-risk@champion`

**Pemilik:** AI/ML Engineer & analis risiko kredit. **Tujuan:** menentukan tindak lanjut berdasarkan bukti, bukan otomatis retrain ketika terjadi drift.

## 1. Pemicu
- Input rusak: error rate **>1%**; operasional: p95 latency **>200 ms**.
- Kualitas model: ROC-AUC **<0,70**, bila label tersedia dan mencakup dua kelas.
- Data drift: PSI **>0,20**, p-value KS **<0,05**, atau JS distance **>0,10**.

## 2. Pemeriksaan 15 menit pertama
1. Buka `reports/monitoring_report.md`, `reports/drift_metrics.csv`, `reports/monitoring_dashboard.png`, dan `logs/predictions.csv`.
2. Konfirmasi periode/batch, jumlah request, jumlah error, p95, metrik drift, data referensi, dan ketersediaan label.
3. Cari sampel `status=error`, cocokkan `request_id`, dan pastikan perhitungan tidak memasukkan input rusak ke data drift.

## 3. Identifikasi penyebab
- **Error input:** pemeriksaan nilai hilang, umur/nominal/tenor tidak valid, perubahan schema; perbaiki validasi dan sumber input.
- **Drift:** periksa perubahan distribusi `age`, `credit_amount`, `duration`, dan `purpose` dibanding training, termasuk kemungkinan perubahan kebijakan akuisisi nasabah. Drift saja tidak membuktikan model gagal.
- **Penurunan performa:** hitung AUC dari request valid **setelah label tersedia**; periksa ukuran sampel serta perbedaan segmen.
- **Latency tinggi:** periksa durasi inferensi per request dan kesehatan mesin; optimalkan serving jika perlu.

## 4. Tindakan
- Perbaiki upstream data/validasi terlebih dahulu bila error rate meningkat.
- Investigasi drift dan pantau kualitas prediksi; **jangan retrain otomatis** hanya dari PSI/KS/JS.
- Pertimbangkan **rollback** jika gangguan terbukti terkait model versi baru dan versi sebelumnya telah teruji lebih aman; pastikan alias MLflow diarahkan ke versi tersebut dan uji ulang serving.

## 5. Retrain & verifikasi
1. Kumpulkan data baru yang bersih **dan sudah memiliki label yang valid**, bukan sekadar trafik belum berlabel.
2. Bangun kandidat menggunakan preprocessing yang sama, split dan audit leakage, catat `data_version`, kemudian jalankan `python src/pipeline.py --reason "investigasi drift"`.
3. Pastikan `python -m pytest -q` lulus, quality gate (AUC ≥0,70; recall bad ≥0,50) lulus, dan AUC kandidat mengungguli champion sebelum promosi. Jika gagal, jangan promosi.

## 6. Catatan insiden
Catat waktu, batch, versi/alias model, request_id relevan, angka sebelum–sesudah, hipotesis, tindakan, hasil verifikasi, dan keputusan. Eskalasi ke AI/ML Engineer, pemilik data, dan analis risiko kredit.
