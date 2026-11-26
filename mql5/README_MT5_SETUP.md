# MetaTrader 5 Zero-DLL Bridge Setup Guide

`MaxTradingBridge.mq5` menggunakan fungsi bawaan MQL5 `WebRequest()` ke `http://127.0.0.1:8765` sehingga **tidak membutuhkan DLL eksternal** dan dilengkapi **Hard Risk Guard independen** langsung di dalam MT5.

## Langkah Instalasi (Akun Demo Terlebih Dahulu!)

1. **Aktifkan WebRequest Allowlist di MT5**:
   - Buka MetaTrader 5 → menu **Tools → Options → Expert Advisors**.
   - Centang **Allow Algorithmic Trading**.
   - Centang **Allow WebRequest for listed URL**, lalu tambahkan:
     `http://127.0.0.1:8765`
   - Klik **OK**.

2. **Install & Compile EA**:
   - Salin file `mql5/MaxTradingBridge.mq5` ke folder `MQL5/Experts/` di Data Folder MT5 (`File → Open Data Folder`).
   - Buka di **MetaEditor** dan tekan **F7 (Compile)** — pastikan `0 errors, 0 warnings`.

3. **Sinkronkan Security Token**:
   - Salin `.env.example` menjadi `.env` lalu set `BRIDGE_AUTH_TOKEN` dengan token acak 32+ karakter.
   - Jalankan Python Brain:
     ```powershell
     python -m backend.main
     ```

4. **Pasang EA di Chart (Cukup 1 Chart, Multi-Symbol Otomatis!)**:
   - Drag `MaxTradingBridge` ke chart `XAUUSD` timeframe **M1**.
   - Pada tab **Inputs**:
     - `BridgeAuthToken`: samakan persis dengan `BRIDGE_AUTH_TOKEN` di `.env`.
     - `SymbolSuffix`: isi jika broker memakai suffix (contoh `m` untuk `XAUUSDm`, atau `.pro` untuk `XAUUSD.pro`, kosongkan jika standar).
     - Verifikasi batas pengaman hardware (`HardMaxDailyDDPct`, `HardMaxLot`, `HardMaxPositions`).
   - Klik **OK** dan pastikan ikon **Algo Trading** di toolbar MT5 menyala hijau.

5. **Verifikasi di Web Dashboard**:
   - Buka `http://127.0.0.1:8765` → pastikan indikator top bar menunjukkan **`BRIDGE`** hijau dengan latency < 80ms dan tab **Risk, Shields & MT5** menampilkan `SymbolSpec` broker Anda.
