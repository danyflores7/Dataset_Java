# Reporte de Validación QA del Dataset de Clones

## 1. Conteos Globales (Estadísticas)

| Etiqueta (clone_type) | Cantidad de Pares |
|-----------------------|-------------------|
| MT3 | 170,053 |
| ST3 | 9,711 |
| T0 | 1,170,339 |
| T1 | 1,531 |
| T2 | 342 |
| T4 | 7,499 |
| VST3 | 672 |
| WT3 | 379,212 |
| **TOTAL GLOBAL** | **1,739,359** |


## 2. Muestreo Aleatorio (Extracción Manual)

### Muestras de Tipo T0

#### Muestra 1 (id1: `14194234` | id2: `4164833`)

**Fragmento 1 (`func1`):**
```java
private String File2String(String directory, String filename) {
        String line;
        InputStream in = null;
        try {
            File f = new File(filename);
            System.out.println("File On:>>>>>>>>>> " + f.getCanonicalPath());
            in = new FileInputStream(f);
        } catch (FileNotFoundException ex) {
            in = null;
        } catch (IOException ex) {
            in = null;
        }
        try {
            if (in == null) {
                filename = directory + "/" + filename;
                java.net.URL urlFile = ClassLoader.getSystemResource(filename);
                if (urlFile == null) {
                    System.out.println("Integrated Chips list file not found: " + filename);
                    System.exit(-1);
                }
                in = urlFile.openStream();
            }
            BufferedReader reader = new BufferedReader(new InputStreamReader(in));
            StringBuffer xmlText = new StringBuffer();
            while ((line = reader.readLine()) != null) {
                xmlText.append(line);
            }
            reader.close();
            return xmlText.toString();
        } catch (FileNotFoundException ex) {
            System.out.println("Integrated Chips list file not found");
            System.exit(-1);
        } catch (IOException ex) {
            ex.printStackTrace();
            System.exit(-1);
        }
        return null;
    }
```

**Fragmento 2 (`func2`):**
```java
public static void buildDeb(File debFile, File controlFile, File dataFile) throws IOException {
        long now = new Date().getTime() / 1000;
        OutputStream deb = new FileOutputStream(debFile);
        deb.write("!<arch>\n".getBytes());
        startFileEntry(deb, DEBIAN_BINARY_NAME, now, DEBIAN_BINARY_CONTENT.length());
        deb.write(DEBIAN_BINARY_CONTENT.getBytes());
        endFileEntry(deb, DEBIAN_BINARY_CONTENT.length());
        startFileEntry(deb, CONTROL_NAME, now, controlFile.length());
        FileInputStream control = new FileInputStream(controlFile);
        byte[] buffer = new byte[1024];
        while (true) {
            int read = control.read(buffer);
            if (read == -1) break;
            deb.write(buffer, 0, read);
        }
        control.close();
        endFileEntry(deb, controlFile.length());
        startFileEntry(deb, DATA_NAME, now, dataFile.length());
        FileInputStream data = new FileInputStream(dataFile);
        while (true) {
            int read = data.read(buffer);
            if (read == -1) break;
            deb.write(buffer, 0, read);
        }
        data.close();
        endFileEntry(deb, dataFile.length());
        deb.close();
    }
```

---

#### Muestra 2 (id1: `3463983` | id2: `13510171`)

**Fragmento 1 (`func1`):**
```java
@Override
        public long getLastModified(final Resource arg0) {
            try {
                final ServletContext context = CContext.getInstance().getContext();
                final URL url = context.getResource(arg0.getName());
                final URLConnection conn = url.openConnection();
                final long lm = conn.getLastModified();
                try {
                    conn.getInputStream().close();
                } catch (final Exception ignore) {
                    ;
                }
                return lm;
            } catch (final Exception e) {
                return 0;
            }
        }
```

**Fragmento 2 (`func2`):**
```java
private String readCreditsHtml(IApplication app) {
            final URL url = app.getResources().getCreditsURL();
            StringBuffer buf = new StringBuffer(2048);
            if (url != null) {
                try {
                    BufferedReader rdr = new BufferedReader(new InputStreamReader(url.openStream()));
                    try {
                        String line = null;
                        while ((line = rdr.readLine()) != null) {
                            String internationalizedLine = Utilities.replaceI18NSpanLine(line, s_stringMgr);
                            buf.append(internationalizedLine);
                        }
                    } finally {
                        rdr.close();
                    }
                } catch (IOException ex) {
                    String errorMsg = s_stringMgr.getString("AboutBoxDialog.error.creditsfile");
                    s_log.error(errorMsg, ex);
                    buf.append(errorMsg + ": " + ex.toString());
                }
            } else {
                String errorMsg = s_stringMgr.getString("AboutBoxDialog.error.creditsfileurl");
                s_log.error(errorMsg);
                buf.append(errorMsg);
            }
            return buf.toString();
        }
```

---

#### Muestra 3 (id1: `19309675` | id2: `73657`)

**Fragmento 1 (`func1`):**
```java
public final void propertyChange(final PropertyChangeEvent event) {
        if (fChecker != null && event.getProperty().equals(ISpellCheckPreferenceKeys.SPELLING_USER_DICTIONARY)) {
            if (fUserDictionary != null) {
                fChecker.removeDictionary(fUserDictionary);
                fUserDictionary = null;
            }
            final String file = (String) event.getNewValue();
            if (file.length() > 0) {
                try {
                    final URL url = new URL("file", null, file);
                    InputStream stream = url.openStream();
                    if (stream != null) {
                        try {
                            fUserDictionary = new PersistentSpellDictionary(url);
                            fChecker.addDictionary(fUserDictionary);
                        } finally {
                            stream.close();
                        }
                    }
                } catch (MalformedURLException exception) {
                } catch (IOException exception) {
                }
            }
        }
    }
```

**Fragmento 2 (`func2`):**
```java
public static void test(String args[]) {
        int trace;
        int bytes_read = 0;
        int last_contentLenght = 0;
        try {
            BufferedReader reader;
            URL url;
            url = new URL(args[0]);
            URLConnection istream = url.openConnection();
            last_contentLenght = istream.getContentLength();
            reader = new BufferedReader(new InputStreamReader(istream.getInputStream()));
            System.out.println(url.toString());
            String line;
            trace = t2pNewTrace();
            while ((line = reader.readLine()) != null) {
                bytes_read = bytes_read + line.length() + 1;
                t2pProcessLine(trace, line);
            }
            t2pHandleEventPairs(trace);
            t2pSort(trace, 0);
            t2pExportTrace(trace, new String("pngtest2.png"), 1000, 700, (float) 0, (float) 33);
            t2pExportTrace(trace, new String("pngtest3.png"), 1000, 700, (float) 2.3, (float) 2.44);
            System.out.println("Press any key to contiune read from stream !!!");
            System.out.println(t2pGetProcessName(trace, 0));
            System.in.read();
            istream = url.openConnection();
            if (last_contentLenght != istream.getContentLength()) {
                istream = url.openConnection();
                istream.setRequestProperty("Range", "bytes=" + Integer.toString(bytes_read) + "-");
                System.out.println(Integer.toString(istream.getContentLength()));
                reader = new BufferedReader(new InputStreamReader(istream.getInputStream()));
                while ((line = reader.readLine()) != null) {
                    System.out.println(line);
                    t2pProcessLine(trace, line);
                }
            } else System.out.println("File not changed !");
            t2pDeleteTrace(trace);
        } catch (MalformedURLException e) {
            System.out.println("MalformedURLException !!!");
        } catch (IOException e) {
            System.out.println("File not found " + args[0]);
        }
        ;
    }
```

---

### Muestras de Tipo T1

#### Muestra 1 (id1: `467550` | id2: `340064`)

**Fragmento 1 (`func1`):**
```java
public void convert(File src, File dest) throws IOException {
        InputStream in = new BufferedInputStream(new FileInputStream(src));
        DcmParser p = pfact.newDcmParser(in);
        Dataset ds = fact.newDataset();
        p.setDcmHandler(ds.getDcmHandler());
        try {
            FileFormat format = p.detectFileFormat();
            if (format != FileFormat.ACRNEMA_STREAM) {
                System.out.println("\n" + src + ": not an ACRNEMA stream!");
                return;
            }
            p.parseDcmFile(format, Tags.PixelData);
            if (ds.contains(Tags.StudyInstanceUID) || ds.contains(Tags.SeriesInstanceUID) || ds.contains(Tags.SOPInstanceUID) || ds.contains(Tags.SOPClassUID)) {
                System.out.println("\n" + src + ": contains UIDs!" + " => probable already DICOM - do not convert");
                return;
            }
            boolean hasPixelData = p.getReadTag() == Tags.PixelData;
            boolean inflate = hasPixelData && ds.getInt(Tags.BitsAllocated, 0) == 12;
            int pxlen = p.getReadLength();
            if (hasPixelData) {
                if (inflate) {
                    ds.putUS(Tags.BitsAllocated, 16);
                    pxlen = pxlen * 4 / 3;
                }
                if (pxlen != (ds.getInt(Tags.BitsAllocated, 0) >>> 3) * ds.getInt(Tags.Rows, 0) * ds.getInt(Tags.Columns, 0) * ds.getInt(Tags.NumberOfFrames, 1) * ds.getInt(Tags.NumberOfSamples, 1)) {
                    System.out.println("\n" + src + ": mismatch pixel data length!" + " => do not convert");
                    return;
                }
            }
            ds.putUI(Tags.StudyInstanceUID, uid(studyUID));
            ds.putUI(Tags.SeriesInstanceUID, uid(seriesUID));
            ds.putUI(Tags.SOPInstanceUID, uid(instUID));
            ds.putUI(Tags.SOPClassUID, classUID);
            if (!ds.contains(Tags.NumberOfSamples)) {
                ds.putUS(Tags.NumberOfSamples, 1);
            }
            if (!ds.contains(Tags.PhotometricInterpretation)) {
                ds.putCS(Tags.PhotometricInterpretation, "MONOCHROME2");
            }
            if (fmi) {
                ds.setFileMetaInfo(fact.newFileMetaInfo(ds, UIDs.ImplicitVRLittleEndian));
            }
            OutputStream out = new BufferedOutputStream(new FileOutputStream(dest));
            try {
            } finally {
                ds.writeFile(out, encodeParam());
                if (hasPixelData) {
                    if (!skipGroupLen) {
                        out.write(PXDATA_GROUPLEN);
                        int grlen = pxlen + 8;
                        out.write((byte) grlen);
                        out.write((byte) (grlen >> 8));
                        out.write((byte) (grlen >> 16));
                        out.write((byte) (grlen >> 24));
                    }
                    out.write(PXDATA_TAG);
                    out.write((byte) pxlen);
                    out.write((byte) (pxlen >> 8));
                    out.write((byte) (pxlen >> 16));
                    out.write((byte) (pxlen >> 24));
                }
                if (inflate) {
                    int b2, b3;
                    for (; pxlen > 0; pxlen -= 3) {
                        out.write(in.read());
                        b2 = in.read();
                        b3 = in.read();
                        out.write(b2 & 0x0f);
                        out.write(b2 >> 4 | ((b3 & 0x0f) << 4));
                        out.write(b3 >> 4);
                    }
                } else {
                    for (; pxlen > 0; --pxlen) {
                        out.write(in.read());
                    }
                }
                out.close();
            }
            System.out.print('.');
        } finally {
            in.close();
        }
    }
```

**Fragmento 2 (`func2`):**
```java
public void convert(File src, File dest) throws IOException {
        InputStream in = new BufferedInputStream(new FileInputStream(src));
        DcmParser p = pfact.newDcmParser(in);
        Dataset ds = fact.newDataset();
        p.setDcmHandler(ds.getDcmHandler());
        try {
            FileFormat format = p.detectFileFormat();
            if (format != FileFormat.ACRNEMA_STREAM) {
                System.out.println("\n" + src + ": not an ACRNEMA stream!");
                return;
            }
            p.parseDcmFile(format, Tags.PixelData);
            if (ds.contains(Tags.StudyInstanceUID) || ds.contains(Tags.SeriesInstanceUID) || ds.contains(Tags.SOPInstanceUID) || ds.contains(Tags.SOPClassUID)) {
                System.out.println("\n" + src + ": contains UIDs!" + " => probable already DICOM - do not convert");
                return;
            }
            boolean hasPixelData = p.getReadTag() == Tags.PixelData;
            boolean inflate = hasPixelData && ds.getInt(Tags.BitsAllocated, 0) == 12;
            int pxlen = p.getReadLength();
            if (hasPixelData) {
                if (inflate) {
                    ds.putUS(Tags.BitsAllocated, 16);
                    pxlen = pxlen * 4 / 3;
                }
                if (pxlen != (ds.getInt(Tags.BitsAllocated, 0) >>> 3) * ds.getInt(Tags.Rows, 0) * ds.getInt(Tags.Columns, 0) * ds.getInt(Tags.NumberOfFrames, 1) * ds.getInt(Tags.NumberOfSamples, 1)) {
                    System.out.println("\n" + src + ": mismatch pixel data length!" + " => do not convert");
                    return;
                }
            }
            ds.putUI(Tags.StudyInstanceUID, uid(studyUID));
            ds.putUI(Tags.SeriesInstanceUID, uid(seriesUID));
            ds.putUI(Tags.SOPInstanceUID, uid(instUID));
            ds.putUI(Tags.SOPClassUID, classUID);
            if (!ds.contains(Tags.NumberOfSamples)) {
                ds.putUS(Tags.NumberOfSamples, 1);
            }
            if (!ds.contains(Tags.PhotometricInterpretation)) {
                ds.putCS(Tags.PhotometricInterpretation, "MONOCHROME2");
            }
            if (fmi) {
                ds.setFileMetaInfo(fact.newFileMetaInfo(ds, UIDs.ImplicitVRLittleEndian));
            }
            OutputStream out = new BufferedOutputStream(new FileOutputStream(dest));
            try {
            } finally {
                ds.writeFile(out, encodeParam());
                if (hasPixelData) {
                    if (!skipGroupLen) {
                        out.write(PXDATA_GROUPLEN);
                        int grlen = pxlen + 8;
                        out.write((byte) grlen);
                        out.write((byte) (grlen >> 8));
                        out.write((byte) (grlen >> 16));
                        out.write((byte) (grlen >> 24));
                    }
                    out.write(PXDATA_TAG);
                    out.write((byte) pxlen);
                    out.write((byte) (pxlen >> 8));
                    out.write((byte) (pxlen >> 16));
                    out.write((byte) (pxlen >> 24));
                }
                if (inflate) {
                    int b2, b3;
                    for (; pxlen > 0; pxlen -= 3) {
                        out.write(in.read());
                        b2 = in.read();
                        b3 = in.read();
                        out.write(b2 & 0x0f);
                        out.write(b2 >> 4 | ((b3 & 0x0f) << 4));
                        out.write(b3 >> 4);
                    }
                } else {
                    for (; pxlen > 0; --pxlen) {
                        out.write(in.read());
                    }
                }
                out.close();
            }
            System.out.print('.');
        } finally {
            in.close();
        }
    }
```

---

#### Muestra 2 (id1: `13908` | id2: `506382`)

**Fragmento 1 (`func1`):**
```java
public static void doVersionCheck(View view) {
        view.showWaitCursor();
        try {
            URL url = new URL(jEdit.getProperty("version-check.url"));
            InputStream in = url.openStream();
            BufferedReader bin = new BufferedReader(new InputStreamReader(in));
            String line;
            String develBuild = null;
            String stableBuild = null;
            while ((line = bin.readLine()) != null) {
                if (line.startsWith(".build")) develBuild = line.substring(6).trim(); else if (line.startsWith(".stablebuild")) stableBuild = line.substring(12).trim();
            }
            bin.close();
            if (develBuild != null && stableBuild != null) {
                doVersionCheck(view, stableBuild, develBuild);
            }
        } catch (IOException e) {
            String[] args = { jEdit.getProperty("version-check.url"), e.toString() };
            GUIUtilities.error(view, "read-error", args);
        }
        view.hideWaitCursor();
    }
```

**Fragmento 2 (`func2`):**
```java
public static void doVersionCheck(View view) {
        view.showWaitCursor();
        try {
            URL url = new URL(jEdit.getProperty("version-check.url"));
            InputStream in = url.openStream();
            BufferedReader bin = new BufferedReader(new InputStreamReader(in));
            String line;
            String develBuild = null;
            String stableBuild = null;
            while ((line = bin.readLine()) != null) {
                if (line.startsWith(".build")) develBuild = line.substring(6).trim(); else if (line.startsWith(".stablebuild")) stableBuild = line.substring(12).trim();
            }
            bin.close();
            if (develBuild != null && stableBuild != null) {
                doVersionCheck(view, stableBuild, develBuild);
            }
        } catch (IOException e) {
            String[] args = { jEdit.getProperty("version-check.url"), e.toString() };
            GUIUtilities.error(view, "read-error", args);
        }
        view.hideWaitCursor();
    }
```

---

#### Muestra 3 (id1: `120707` | id2: `621591`)

**Fragmento 1 (`func1`):**
```java
public void convert(File src, File dest) throws IOException {
        InputStream in = new BufferedInputStream(new FileInputStream(src));
        DcmParser p = pfact.newDcmParser(in);
        Dataset ds = fact.newDataset();
        p.setDcmHandler(ds.getDcmHandler());
        try {
            FileFormat format = p.detectFileFormat();
            if (format != FileFormat.ACRNEMA_STREAM) {
                System.out.println("\n" + src + ": not an ACRNEMA stream!");
                return;
            }
            p.parseDcmFile(format, Tags.PixelData);
            if (ds.contains(Tags.StudyInstanceUID) || ds.contains(Tags.SeriesInstanceUID) || ds.contains(Tags.SOPInstanceUID) || ds.contains(Tags.SOPClassUID)) {
                System.out.println("\n" + src + ": contains UIDs!" + " => probable already DICOM - do not convert");
                return;
            }
            boolean hasPixelData = p.getReadTag() == Tags.PixelData;
            boolean inflate = hasPixelData && ds.getInt(Tags.BitsAllocated, 0) == 12;
            int pxlen = p.getReadLength();
            if (hasPixelData) {
                if (inflate) {
                    ds.putUS(Tags.BitsAllocated, 16);
                    pxlen = pxlen * 4 / 3;
                }
                if (pxlen != (ds.getInt(Tags.BitsAllocated, 0) >>> 3) * ds.getInt(Tags.Rows, 0) * ds.getInt(Tags.Columns, 0) * ds.getInt(Tags.NumberOfFrames, 1) * ds.getInt(Tags.NumberOfSamples, 1)) {
                    System.out.println("\n" + src + ": mismatch pixel data length!" + " => do not convert");
                    return;
                }
            }
            ds.putUI(Tags.StudyInstanceUID, uid(studyUID));
            ds.putUI(Tags.SeriesInstanceUID, uid(seriesUID));
            ds.putUI(Tags.SOPInstanceUID, uid(instUID));
            ds.putUI(Tags.SOPClassUID, classUID);
            if (!ds.contains(Tags.NumberOfSamples)) {
                ds.putUS(Tags.NumberOfSamples, 1);
            }
            if (!ds.contains(Tags.PhotometricInterpretation)) {
                ds.putCS(Tags.PhotometricInterpretation, "MONOCHROME2");
            }
            if (fmi) {
                ds.setFileMetaInfo(fact.newFileMetaInfo(ds, UIDs.ImplicitVRLittleEndian));
            }
            OutputStream out = new BufferedOutputStream(new FileOutputStream(dest));
            try {
            } finally {
                ds.writeFile(out, encodeParam());
                if (hasPixelData) {
                    if (!skipGroupLen) {
                        out.write(PXDATA_GROUPLEN);
                        int grlen = pxlen + 8;
                        out.write((byte) grlen);
                        out.write((byte) (grlen >> 8));
                        out.write((byte) (grlen >> 16));
                        out.write((byte) (grlen >> 24));
                    }
                    out.write(PXDATA_TAG);
                    out.write((byte) pxlen);
                    out.write((byte) (pxlen >> 8));
                    out.write((byte) (pxlen >> 16));
                    out.write((byte) (pxlen >> 24));
                }
                if (inflate) {
                    int b2, b3;
                    for (; pxlen > 0; pxlen -= 3) {
                        out.write(in.read());
                        b2 = in.read();
                        b3 = in.read();
                        out.write(b2 & 0x0f);
                        out.write(b2 >> 4 | ((b3 & 0x0f) << 4));
                        out.write(b3 >> 4);
                    }
                } else {
                    for (; pxlen > 0; --pxlen) {
                        out.write(in.read());
                    }
                }
                out.close();
            }
            System.out.print('.');
        } finally {
            in.close();
        }
    }
```

**Fragmento 2 (`func2`):**
```java
public void convert(File src, File dest) throws IOException {
        InputStream in = new BufferedInputStream(new FileInputStream(src));
        DcmParser p = pfact.newDcmParser(in);
        Dataset ds = fact.newDataset();
        p.setDcmHandler(ds.getDcmHandler());
        try {
            FileFormat format = p.detectFileFormat();
            if (format != FileFormat.ACRNEMA_STREAM) {
                System.out.println("\n" + src + ": not an ACRNEMA stream!");
                return;
            }
            p.parseDcmFile(format, Tags.PixelData);
            if (ds.contains(Tags.StudyInstanceUID) || ds.contains(Tags.SeriesInstanceUID) || ds.contains(Tags.SOPInstanceUID) || ds.contains(Tags.SOPClassUID)) {
                System.out.println("\n" + src + ": contains UIDs!" + " => probable already DICOM - do not convert");
                return;
            }
            boolean hasPixelData = p.getReadTag() == Tags.PixelData;
            boolean inflate = hasPixelData && ds.getInt(Tags.BitsAllocated, 0) == 12;
            int pxlen = p.getReadLength();
            if (hasPixelData) {
                if (inflate) {
                    ds.putUS(Tags.BitsAllocated, 16);
                    pxlen = pxlen * 4 / 3;
                }
                if (pxlen != (ds.getInt(Tags.BitsAllocated, 0) >>> 3) * ds.getInt(Tags.Rows, 0) * ds.getInt(Tags.Columns, 0) * ds.getInt(Tags.NumberOfFrames, 1) * ds.getInt(Tags.NumberOfSamples, 1)) {
                    System.out.println("\n" + src + ": mismatch pixel data length!" + " => do not convert");
                    return;
                }
            }
            ds.putUI(Tags.StudyInstanceUID, uid(studyUID));
            ds.putUI(Tags.SeriesInstanceUID, uid(seriesUID));
            ds.putUI(Tags.SOPInstanceUID, uid(instUID));
            ds.putUI(Tags.SOPClassUID, classUID);
            if (!ds.contains(Tags.NumberOfSamples)) {
                ds.putUS(Tags.NumberOfSamples, 1);
            }
            if (!ds.contains(Tags.PhotometricInterpretation)) {
                ds.putCS(Tags.PhotometricInterpretation, "MONOCHROME2");
            }
            if (fmi) {
                ds.setFileMetaInfo(fact.newFileMetaInfo(ds, UIDs.ImplicitVRLittleEndian));
            }
            OutputStream out = new BufferedOutputStream(new FileOutputStream(dest));
            try {
            } finally {
                ds.writeFile(out, encodeParam());
                if (hasPixelData) {
                    if (!skipGroupLen) {
                        out.write(PXDATA_GROUPLEN);
                        int grlen = pxlen + 8;
                        out.write((byte) grlen);
                        out.write((byte) (grlen >> 8));
                        out.write((byte) (grlen >> 16));
                        out.write((byte) (grlen >> 24));
                    }
                    out.write(PXDATA_TAG);
                    out.write((byte) pxlen);
                    out.write((byte) (pxlen >> 8));
                    out.write((byte) (pxlen >> 16));
                    out.write((byte) (pxlen >> 24));
                }
                if (inflate) {
                    int b2, b3;
                    for (; pxlen > 0; pxlen -= 3) {
                        out.write(in.read());
                        b2 = in.read();
                        b3 = in.read();
                        out.write(b2 & 0x0f);
                        out.write(b2 >> 4 | ((b3 & 0x0f) << 4));
                        out.write(b3 >> 4);
                    }
                } else {
                    for (; pxlen > 0; --pxlen) {
                        out.write(in.read());
                    }
                }
                out.close();
            }
            System.out.print('.');
        } finally {
            in.close();
        }
    }
```

---

### Muestras de Tipo T2

#### Muestra 1 (id1: `8793825` | id2: `1421558`)

**Fragmento 1 (`func1`):**
```java
public static boolean encodeFileToFile(String infile, String outfile) {
        boolean success = false;
        java.io.InputStream in = null;
        java.io.OutputStream out = null;
        try {
            in = new Base64.InputStream(new java.io.BufferedInputStream(new java.io.FileInputStream(infile)), Base64.ENCODE);
            out = new java.io.BufferedOutputStream(new java.io.FileOutputStream(outfile));
            byte[] buffer = new byte[65536];
            int read = -1;
            while ((read = in.read(buffer)) >= 0) {
                out.write(buffer, 0, read);
            }
            success = true;
        } catch (java.io.IOException exc) {
            exc.printStackTrace();
        } finally {
            try {
                in.close();
            } catch (Exception exc) {
            }
            try {
                out.close();
            } catch (Exception exc) {
            }
        }
        return success;
    }
```

**Fragmento 2 (`func2`):**
```java
public static boolean decodeFileToFile(String infile, String outfile) {
        boolean success = false;
        java.io.InputStream in = null;
        java.io.OutputStream out = null;
        try {
            in = new Base64.InputStream(new java.io.BufferedInputStream(new java.io.FileInputStream(infile)), Base64.DECODE);
            out = new java.io.BufferedOutputStream(new java.io.FileOutputStream(outfile));
            byte[] buffer = new byte[65536];
            int read = -1;
            while ((read = in.read(buffer)) >= 0) {
                out.write(buffer, 0, read);
            }
            success = true;
        } catch (java.io.IOException exc) {
            exc.printStackTrace();
        } finally {
            try {
                in.close();
            } catch (Exception exc) {
            }
            try {
                out.close();
            } catch (Exception exc) {
            }
        }
        return success;
    }
```

---

#### Muestra 2 (id1: `12678589` | id2: `18269744`)

**Fragmento 1 (`func1`):**
```java
public static boolean decodeFileToFile(String infile, String outfile) {
        boolean success = false;
        java.io.InputStream in = null;
        java.io.OutputStream out = null;
        try {
            in = new Base64.InputStream(new java.io.BufferedInputStream(new java.io.FileInputStream(infile)), Base64.DECODE);
            out = new java.io.BufferedOutputStream(new java.io.FileOutputStream(outfile));
            byte[] buffer = new byte[65536];
            int read = -1;
            while ((read = in.read(buffer)) >= 0) {
                out.write(buffer, 0, read);
            }
            success = true;
        } catch (java.io.IOException exc) {
            exc.printStackTrace();
        } finally {
            try {
                in.close();
            } catch (Exception exc) {
            }
            try {
                out.close();
            } catch (Exception exc) {
            }
        }
        return success;
    }
```

**Fragmento 2 (`func2`):**
```java
public static boolean encodeFileToFile(String infile, String outfile) {
        boolean success = false;
        java.io.InputStream in = null;
        java.io.OutputStream out = null;
        try {
            in = new Base64.InputStream(new java.io.BufferedInputStream(new java.io.FileInputStream(infile)), Base64.ENCODE);
            out = new java.io.BufferedOutputStream(new java.io.FileOutputStream(outfile));
            byte[] buffer = new byte[65536];
            int read = -1;
            while ((read = in.read(buffer)) >= 0) {
                out.write(buffer, 0, read);
            }
            success = true;
        } catch (java.io.IOException exc) {
            exc.printStackTrace();
        } finally {
            try {
                in.close();
            } catch (Exception exc) {
            }
            try {
                out.close();
            } catch (Exception exc) {
            }
        }
        return success;
    }
```

---

#### Muestra 3 (id1: `13783898` | id2: `8778962`)

**Fragmento 1 (`func1`):**
```java
public static boolean encodeFileToFile(String infile, String outfile) {
        boolean success = false;
        java.io.InputStream in = null;
        java.io.OutputStream out = null;
        try {
            in = new Base64.InputStream(new java.io.BufferedInputStream(new java.io.FileInputStream(infile)), Base64.ENCODE);
            out = new java.io.BufferedOutputStream(new java.io.FileOutputStream(outfile));
            byte[] buffer = new byte[65536];
            int read = -1;
            while ((read = in.read(buffer)) >= 0) {
                out.write(buffer, 0, read);
            }
            success = true;
        } catch (java.io.IOException exc) {
            exc.printStackTrace();
        } finally {
            try {
                in.close();
            } catch (Exception exc) {
            }
            try {
                out.close();
            } catch (Exception exc) {
            }
        }
        return success;
    }
```

**Fragmento 2 (`func2`):**
```java
public static boolean decodeFileToFile(String infile, String outfile) {
        boolean success = false;
        java.io.InputStream in = null;
        java.io.OutputStream out = null;
        try {
            in = new Base64.InputStream(new java.io.BufferedInputStream(new java.io.FileInputStream(infile)), Base64.DECODE);
            out = new java.io.BufferedOutputStream(new java.io.FileOutputStream(outfile));
            byte[] buffer = new byte[65536];
            int read = -1;
            while ((read = in.read(buffer)) >= 0) {
                out.write(buffer, 0, read);
            }
            success = true;
        } catch (java.io.IOException exc) {
            exc.printStackTrace();
        } finally {
            try {
                in.close();
            } catch (Exception exc) {
            }
            try {
                out.close();
            } catch (Exception exc) {
            }
        }
        return success;
    }
```

---

### Muestras de Tipo VST3

#### Muestra 1 (id1: `3365958` | id2: `11892804`)

**Fragmento 1 (`func1`):**
```java
private void bubbleSort(int[] mas) {
        boolean t = true;
        int temp = 0;
        while (t) {
            t = false;
            for (int i = 0; i < mas.length - 1; i++) {
                if (mas[i] > mas[i + 1]) {
                    temp = mas[i];
                    mas[i] = mas[i + 1];
                    mas[i + 1] = temp;
                    t = true;
                }
            }
        }
    }
```

**Fragmento 2 (`func2`):**
```java
private int[] Tri(int[] pertinence, int taille) {
        boolean change = true;
        int tmp;
        while (change) {
            change = false;
            for (int i = 0; i < taille - 2; i++) {
                if (pertinence[i] < pertinence[i + 1]) {
                    tmp = pertinence[i];
                    pertinence[i] = pertinence[i + 1];
                    pertinence[i + 1] = tmp;
                    change = true;
                }
            }
        }
        return pertinence;
    }
```

---

#### Muestra 2 (id1: `14230405` | id2: `23058336`)

**Fragmento 1 (`func1`):**
```java
public void copyFile(String from, String to) throws IOException {
        FileChannel srcChannel = new FileInputStream(from).getChannel();
        FileChannel dstChannel = new FileOutputStream(to).getChannel();
        dstChannel.transferFrom(srcChannel, 0, srcChannel.size());
        srcChannel.close();
        dstChannel.close();
    }
```

**Fragmento 2 (`func2`):**
```java
public static void copyFile(File in, File out) throws Exception {
        FileChannel sourceChannel = new FileInputStream(in).getChannel();
        FileChannel destinationChannel = new FileOutputStream(out).getChannel();
        sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
        sourceChannel.close();
        destinationChannel.close();
    }
```

---

#### Muestra 3 (id1: `20082006` | id2: `10656323`)

**Fragmento 1 (`func1`):**
```java
public static void copyFile(File src, File dst) throws IOException {
        FileChannel sourceChannel = new FileInputStream(src).getChannel();
        FileChannel destinationChannel = new FileOutputStream(dst).getChannel();
        sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
        sourceChannel.close();
        destinationChannel.close();
    }
```

**Fragmento 2 (`func2`):**
```java
public void copy(File s, File t) throws IOException {
            FileChannel in = (new FileInputStream(s)).getChannel();
            FileChannel out = (new FileOutputStream(t)).getChannel();
            in.transferTo(0, s.length(), out);
            in.close();
            out.close();
        }
```

---

### Muestras de Tipo ST3

#### Muestra 1 (id1: `6190168` | id2: `3024993`)

**Fragmento 1 (`func1`):**
```java
public void xtest12() throws Exception {
        PDFManager manager = new ITextManager();
        InputStream pdf = new FileInputStream("/tmp/090237098008f637.pdf");
        InputStream page1 = manager.cut(pdf, 36, 36);
        OutputStream outputStream = new FileOutputStream("/tmp/090237098008f637-1.pdf");
        IOUtils.copy(page1, outputStream);
        outputStream.close();
        pdf.close();
    }
```

**Fragmento 2 (`func2`):**
```java
@Test
    public void testCopy_readerToWriter_nullOut() throws Exception {
        InputStream in = new ByteArrayInputStream(inData);
        in = new YellOnCloseInputStreamTest(in);
        Reader reader = new InputStreamReader(in, "US-ASCII");
        try {
            IOUtils.copy(reader, (Writer) null);
            fail();
        } catch (NullPointerException ex) {
        }
    }
```

---

#### Muestra 2 (id1: `8000624` | id2: `19849797`)

**Fragmento 1 (`func1`):**
```java
private void CopyTo(File dest) throws IOException {
        FileReader in = null;
        FileWriter out = null;
        int c;
        try {
            in = new FileReader(image);
            out = new FileWriter(dest);
            while ((c = in.read()) != -1) out.write(c);
        } finally {
            if (in != null) try {
                in.close();
            } catch (Exception e) {
            }
            if (out != null) try {
                out.close();
            } catch (Exception e) {
            }
        }
    }
```

**Fragmento 2 (`func2`):**
```java
public static void copyFile(File sourceFile, File destFile) throws IOException {
        if (!destFile.exists()) {
            destFile.createNewFile();
        }
        FileChannel source = null;
        FileChannel destination = null;
        try {
            source = new FileInputStream(sourceFile).getChannel();
            destination = new FileOutputStream(destFile).getChannel();
            destination.transferFrom(source, 0, source.size());
        } finally {
            if (source != null) {
                source.close();
            }
            if (destination != null) {
                destination.close();
            }
        }
    }
```

---

#### Muestra 3 (id1: `23658098` | id2: `14729256`)

**Fragmento 1 (`func1`):**
```java
public void copyFile(File in, File out) throws Exception {
        FileChannel sourceChannel = new FileInputStream(in).getChannel();
        FileChannel destinationChannel = new FileOutputStream(out).getChannel();
        sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
        sourceChannel.close();
        destinationChannel.close();
    }
```

**Fragmento 2 (`func2`):**
```java
public static void copyFile(File file, String pathExport) throws IOException {
        File out = new File(pathExport);
        FileChannel sourceChannel = new FileInputStream(file).getChannel();
        FileChannel destinationChannel = new FileOutputStream(out).getChannel();
        sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
        sourceChannel.close();
        destinationChannel.close();
    }
```

---

### Muestras de Tipo MT3

#### Muestra 1 (id1: `3621790` | id2: `11507466`)

**Fragmento 1 (`func1`):**
```java
private String sha1(String s) {
        String encrypt = s;
        try {
            MessageDigest sha = MessageDigest.getInstance("SHA-1");
            sha.update(s.getBytes());
            byte[] digest = sha.digest();
            final StringBuffer buffer = new StringBuffer();
            for (int i = 0; i < digest.length; ++i) {
                final byte b = digest[i];
                final int value = (b & 0x7F) + (b < 0 ? 128 : 0);
                buffer.append(value < 16 ? "0" : "");
                buffer.append(Integer.toHexString(value));
            }
            encrypt = buffer.toString();
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }
        return encrypt;
    }
```

**Fragmento 2 (`func2`):**
```java
public PollSetMessage(String username, String question, String title, String[] choices) {
        this.username = username;
        MessageDigest m = null;
        try {
            m = MessageDigest.getInstance("SHA-1");
        } catch (NoSuchAlgorithmException ex) {
            ex.printStackTrace();
        }
        String id = username + String.valueOf(System.nanoTime());
        m.update(id.getBytes(), 0, id.length());
        voteId = new BigInteger(1, m.digest()).toString(16);
        this.question = question;
        this.title = title;
        this.choices = choices;
    }
```

---

#### Muestra 2 (id1: `1760354` | id2: `5977092`)

**Fragmento 1 (`func1`):**
```java
public static int deleteOrderStatusHis(String likePatten) {
        Connection conn = null;
        PreparedStatement psmt = null;
        StringBuffer SQL = new StringBuffer(200);
        int deleted = 0;
        SQL.append(" DELETE FROM JHF_ORDER_STATUS_HISTORY ").append(" WHERE   ORDER_ID LIKE  ? ");
        try {
            conn = JdbcConnectionPool.mainConnection();
            conn.setAutoCommit(false);
            conn.setReadOnly(false);
            psmt = conn.prepareStatement(SQL.toString());
            psmt.setString(1, "%" + likePatten + "%");
            deleted = psmt.executeUpdate();
            conn.commit();
        } catch (SQLException e) {
            if (null != conn) {
                try {
                    conn.rollback();
                } catch (SQLException e1) {
                    System.out.println(" error when roll back !");
                }
            }
        } finally {
            try {
                if (null != psmt) {
                    psmt.close();
                    psmt = null;
                }
                if (null != conn) {
                    conn.close();
                    conn = null;
                }
            } catch (SQLException e) {
                System.out.println(" error  when psmt close or conn close .");
            }
        }
        return deleted;
    }
```

**Fragmento 2 (`func2`):**
```java
public long create(long mimeTypeId, long sanId) throws SQLException {
        long fileId = 0;
        DataSource ds = getDataSource(DEFAULT_DATASOURCE);
        Connection conn = ds.getConnection();
        try {
            conn.setAutoCommit(false);
            Statement stmt = conn.createStatement();
            stmt.execute(NEXT_FILE_ID);
            ResultSet rs = stmt.getResultSet();
            while (rs.next()) {
                fileId = rs.getLong(NEXTVAL);
            }
            PreparedStatement pstmt = conn.prepareStatement(INSERT_FILE);
            pstmt.setLong(1, fileId);
            pstmt.setLong(2, mimeTypeId);
            pstmt.setLong(3, sanId);
            pstmt.setLong(4, WORKFLOW_ATTENTE_VALIDATION);
            int nbrow = pstmt.executeUpdate();
            if (nbrow == 0) {
                throw new SQLException();
            }
            conn.commit();
            closeRessources(conn, pstmt);
        } catch (SQLException e) {
            log.error("Can't FileDAOImpl.create " + e.getMessage());
            conn.rollback();
            throw e;
        }
        return fileId;
    }
```

---

#### Muestra 3 (id1: `4593011` | id2: `4593012`)

**Fragmento 1 (`func1`):**
```java
private void loadDDL() throws IOException {
        try {
            conn.createStatement().executeQuery("SELECT * FROM overrides").close();
        } catch (SQLException e) {
            Statement stmt = null;
            if (!e.getMessage().startsWith(ERR_MISSING_TABLE)) {
                LOG.fatal(SQL_ERROR, e);
                throw new IOException("Error on initial data store read", e);
            }
            String[] qry = { "CREATE TABLE monitor (id INTEGER PRIMARY KEY NOT NULL, status VARCHAR(32) NOT NULL, next_update TIMESTAMP NOT NULL)", "CREATE TABLE overrides (id INT NOT NULL, title VARCHAR(255) NOT NULL, subtitle VARCHAR(255) NOT NULL, enable BOOLEAN NOT NULL DEFAULT TRUE, PRIMARY KEY(id))", "CREATE TABLE settings (var VARCHAR(32) NOT NULL, val VARCHAR(255) NOT NULL, PRIMARY KEY(var))", "INSERT INTO settings (var, val) VALUES ('schema', '1')" };
            try {
                conn.setAutoCommit(false);
                stmt = conn.createStatement();
                for (String q : qry) stmt.executeUpdate(q);
                conn.commit();
            } catch (SQLException e2) {
                try {
                    conn.rollback();
                } catch (SQLException e3) {
                    LOG.fatal(SQL_ERROR, e3);
                }
                LOG.fatal(SQL_ERROR, e2);
                throw new IOException("Error initializing data store", e2);
            } finally {
                if (stmt != null) {
                    try {
                        stmt.close();
                    } catch (SQLException e4) {
                        LOG.fatal(SQL_ERROR, e4);
                        throw new IOException("Unable to cleanup data store resources", e4);
                    }
                }
                try {
                    conn.setAutoCommit(true);
                } catch (SQLException e3) {
                    LOG.fatal(SQL_ERROR, e3);
                    throw new IOException("Unable to reset data store auto commit", e3);
                }
            }
        }
        return;
    }
```

**Fragmento 2 (`func2`):**
```java
private void upgradeSchema() throws IOException {
        Statement stmt = null;
        try {
            int i = getSchema();
            LOG.info("DB is currently at schema " + i);
            if (i < SCHEMA_VERSION) {
                LOG.info("Upgrading from schema " + i + " to schema " + SCHEMA_VERSION);
                conn.setAutoCommit(false);
                stmt = conn.createStatement();
                while (i < SCHEMA_VERSION) {
                    String qry;
                    switch(i) {
                        case 1:
                            qry = "UPDATE settings SET val = '2' WHERE var = 'schema'";
                            stmt.executeUpdate(qry);
                            break;
                    }
                    i++;
                }
                conn.commit();
            }
        } catch (SQLException e) {
            try {
                conn.rollback();
            } catch (SQLException e2) {
                LOG.error(SQL_ERROR, e2);
            }
            LOG.fatal(SQL_ERROR, e);
            throw new IOException("Error upgrading data store", e);
        } finally {
            try {
                if (stmt != null) stmt.close();
                conn.setAutoCommit(true);
            } catch (SQLException e) {
                LOG.error(SQL_ERROR, e);
                throw new IOException("Unable to cleanup SQL resources", e);
            }
        }
    }
```

---

### Muestras de Tipo T4

#### Muestra 1 (id1: `GPT_T4_prompt_2_2147_1` | id2: `GPT_T4_prompt_2_2147_2`)

**Fragmento 1 (`func1`):**
```java
public static String join (String separator, String...values) {
    if (values.length == 0) return "";
    char [] sep = separator.toCharArray ();
    int totalSize = (values.length - 1) * sep.length;
    for (int i = 0;
    i < values.length; i ++) {
        if (values [i] == null) values [i] = "";
        else totalSize += values [i].length ();
    }
    char [] joined = new char [totalSize];
    int pos = 0;
    for (int i = 0, end = values.length - 1;
    i < end; i ++) {
        System.arraycopy (values [i].toCharArray (), 0, joined, pos, values [i].length ());
        pos += values [i].length ();
        System.arraycopy (sep, 0, joined, pos, sep.length);
        pos += sep.length;
    }
    System.arraycopy (values [values.length - 1].toCharArray (), 0, joined, pos, values [values.length - 1].length ());
    return new String (joined);
}
```

**Fragmento 2 (`func2`):**
```java
public static String join (String delimiter, String...values) 
{ 
    if (values.length == 0) return ""; 
    
     StringBuilder sb = new StringBuilder(); 
    
    for (int i = 0; i < values.length - 1; i++) { 
        sb.append(values[i]).append(delimiter); 
     } 
    
     sb.append(values[values.length - 1]); 
    
     return sb.toString(); 
}
```

---

#### Muestra 2 (id1: `GPT_T4_prompt_2_2432_1` | id2: `GPT_T4_prompt_2_2432_2`)

**Fragmento 1 (`func1`):**
```java
public int solution (final int X, final int [] A) {
    Set < Integer > emptyPosition = new HashSet < Integer > ();
    for (int i = 1;
    i <= X; i ++) {
        emptyPosition.add (i);
    }
    for (int i = 0;
    i < A.length; i ++) {
        emptyPosition.remove (A [i]);
        if (emptyPosition.size () == 0) {
            return i;
        }
    }
    return - 1;
}
```

**Fragmento 2 (`func2`):**
```java
public int solution (final int X, final int [] A) {
    int[] sumArray = new int[X];
    for (int i = 0; i < A.length; i++) {
        if (A[i] <= X) {
            sumArray[A[i] - 1] = 1;
        }
    }
    int currSum = 0;
    for (int i = 0; i < X; i++) {
        currSum += sumArray[i];
        if (currSum == X) {
            return i;
        }
    }
    return -1;
}
```

---

#### Muestra 3 (id1: `GPT_T4_prompt_2_4407_1` | id2: `GPT_T4_prompt_2_4407_2`)

**Fragmento 1 (`func1`):**
```java
protected void layoutPlotChildren () {
    super.layoutPlotChildren ();
    for (Series < String, Number > series : getData ()) {
        for (Data < String, Number > data : series.getData ()) {
            StackPane bar = (StackPane) data.getNode ();
            Label label = null;
            for (Node node : bar.getChildrenUnmodifiable ()) {
                LOGGER.debug ("Bar has child {}, {}.", node, node.getClass ());
                if (node instanceof Label) {
                    label = (Label) node;
                    break;
                }
            }
            if (label == null) {
                label = new Label (series.getName ());
                label.setRotate (90.0);
                bar.getChildren ().add (label);
            } else {
                label.setText (series.getName ());
            }
        }
    }
}
```

**Fragmento 2 (`func2`):**
```java
protected void layoutPlotChildren () {
    super.layoutPlotChildren ();
    getData ().forEach(series -> {
        series.getData ().forEach(data -> {
            StackPane bar = (StackPane) data.getNode ();
            Label label = findLabel(bar);
            if (label == null) {
                bar.getChildren ().add (createLabel (series.getName (), 90.0));
            } else {
                label.setText (series.getName ());
            }
        });
    });
}
```

---

### Muestras de Tipo WT3

#### Muestra 1 (id1: `14956125` | id2: `6416201`)

**Fragmento 1 (`func1`):**
```java
protected String getLibJSCode() throws IOException {
        if (cachedLibJSCode == null) {
            InputStream is = getClass().getResourceAsStream(JS_LIB_FILE);
            StringWriter output = new StringWriter();
            IOUtils.copy(is, output);
            cachedLibJSCode = output.toString();
        }
        return cachedLibJSCode;
    }
```

**Fragmento 2 (`func2`):**
```java
@Override
    public void copyTo(ManagedFile other) throws ManagedIOException {
        try {
            if (other.getType() == ManagedFileType.FILE) {
                IOUtils.copy(this.getContent().getInputStream(), other.getContent().getOutputStream());
            } else {
                ManagedFile newFile = other.retrive(this.getPath());
                newFile.createFile();
                IOUtils.copy(this.getContent().getInputStream(), newFile.getContent().getOutputStream());
            }
        } catch (IOException ioe) {
            throw ManagedIOException.manage(ioe);
        }
    }
```

---

#### Muestra 2 (id1: `19416288` | id2: `4937926`)

**Fragmento 1 (`func1`):**
```java
private void zip(String object, TupleOutput output) {
            byte array[] = object.getBytes();
            try {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                GZIPOutputStream out = new GZIPOutputStream(baos);
                ByteArrayInputStream in = new ByteArrayInputStream(array);
                IOUtils.copyTo(in, out);
                in.close();
                out.close();
                byte array2[] = baos.toByteArray();
                if (array2.length + 4 < array.length) {
                    output.writeBoolean(true);
                    output.writeInt(array2.length);
                    output.write(array2);
                } else {
                    output.writeBoolean(false);
                    output.writeString(object);
                }
            } catch (IOException err) {
                throw new RuntimeException(err);
            }
        }
```

**Fragmento 2 (`func2`):**
```java
@Override
    public RServiceResponse execute(final NexusServiceRequest inData) throws NexusServiceException {
        final RServiceRequest data = (RServiceRequest) inData;
        final RServiceResponse retval = new RServiceResponse();
        final StringBuilder result = new StringBuilder("R service call results:\n");
        RSession session;
        RConnection connection = null;
        try {
            result.append("Session Attachment: \n");
            final byte[] sessionBytes = data.getSession();
            if (sessionBytes != null && sessionBytes.length > 0) {
                session = RUtils.getInstance().bytesToSession(sessionBytes);
                result.append("  attaching to " + session + "\n");
                connection = session.attach();
            } else {
                result.append("  creating new session\n");
                connection = new RConnection(data.getServerAddress());
            }
            result.append("Input Parameters: \n");
            for (String attributeName : data.getInputVariables().keySet()) {
                final Object parameter = data.getInputVariables().get(attributeName);
                if (parameter instanceof URI) {
                    final FileObject file = VFS.getManager().resolveFile(((URI) parameter).toString());
                    final RFileOutputStream ros = connection.createFile(file.getName().getBaseName());
                    IOUtils.copy(file.getContent().getInputStream(), ros);
                    connection.assign(attributeName, file.getName().getBaseName());
                } else {
                    connection.assign(attributeName, RUtils.getInstance().convertToREXP(parameter));
                }
                result.append("  " + parameter.getClass().getSimpleName() + " " + attributeName + "=" + parameter + "\n");
            }
            final REXP rExpression = connection.eval(RUtils.getInstance().wrapCode(data.getCode().replace('\r', '\n')));
            result.append("Execution results:\n" + rExpression.asString() + "\n");
            if (rExpression.isNull() || rExpression.asString().startsWith("Error")) {
                retval.setErr(rExpression.asString());
                throw new NexusServiceException("R error: " + rExpression.asString());
            }
            result.append("Output Parameters:\n");
            final String[] rVariables = connection.eval("ls();").asStrings();
            for (String varname : rVariables) {
                final String[] rVariable = connection.eval("class(" + varname + ")").asStrings();
                if (rVariable.length == 2 && "file".equals(rVariable[0]) && "connection".equals(rVariable[1])) {
                    final String rFileName = connection.eval("showConnections(TRUE)[" + varname + "]").asString();
                    result.append("  R File ").append(varname).append('=').append(rFileName).append('\n');
                    final RFileInputStream rInputStream = connection.openFile(rFileName);
                    final File file = File.createTempFile("nexus-" + data.getRequestId(), ".dat");
                    IOUtils.copy(rInputStream, new FileOutputStream(file));
                    retval.getOutputVariables().put(varname, file.getCanonicalFile().toURI());
                } else {
                    final Object varvalue = RUtils.getInstance().convertREXP(connection.eval(varname));
                    retval.getOutputVariables().put(varname, varvalue);
                    final String printValue = varvalue == null ? "null" : varvalue.getClass().isArray() ? Arrays.asList(varvalue).toString() : varvalue.toString();
                    result.append("  ").append(varvalue == null ? "" : varvalue.getClass().getSimpleName()).append(' ').append(varname).append('=').append(printValue).append('\n');
                }
            }
        } catch (ClassNotFoundException cnfe) {
            retval.setErr(cnfe.getMessage());
            LOGGER.error("Rserve Exception", cnfe);
        } catch (RserveException rse) {
            retval.setErr(rse.getMessage());
            LOGGER.error("Rserve Exception", rse);
        } catch (REXPMismatchException rme) {
            retval.setErr(rme.getMessage());
            LOGGER.error("REXP Mismatch Exception", rme);
        } catch (IOException rme) {
            retval.setErr(rme.getMessage());
            LOGGER.error("IO Exception copying file ", rme);
        } finally {
            result.append("Session Detachment:\n");
            if (connection != null) {
                RSession outSession;
                if (retval.isKeepSession()) {
                    try {
                        outSession = connection.detach();
                    } catch (RserveException e) {
                        LOGGER.debug("Error detaching R session", e);
                        outSession = null;
                    }
                } else {
                    outSession = null;
                }
                final boolean close = outSession == null;
                if (!close) {
                    retval.setSession(RUtils.getInstance().sessionToBytes(outSession));
                    result.append("  suspended session for later use\n");
                }
                connection.close();
                retval.setSession(null);
                result.append("  session closed.\n");
            }
        }
        retval.setOut(result.toString());
        return retval;
    }
```

---

#### Muestra 3 (id1: `11149084` | id2: `9240872`)

**Fragmento 1 (`func1`):**
```java
private void deleteProject(String uid, String home, HttpServletRequest request, HttpServletResponse response) throws Exception {
        String project = request.getParameter("project");
        String line;
        response.setContentType("text/html");
        PrintWriter out = response.getWriter();
        htmlHeader(out, "Project Status", "");
        try {
            synchronized (Class.forName("com.sun.gep.SunTCP")) {
                Vector list = new Vector();
                String directory = home;
                Runtime.getRuntime().exec("/usr/bin/rm -rf " + directory + project);
                FilePermission perm = new FilePermission(directory + SUNTCP_LIST, "read,write,execute");
                File listfile = new File(directory + SUNTCP_LIST);
                BufferedReader read = new BufferedReader(new FileReader(listfile));
                while ((line = read.readLine()) != null) {
                    if (!((new StringTokenizer(line, "\t")).nextToken().equals(project))) {
                        list.addElement(line);
                    }
                }
                read.close();
                if (list.size() > 0) {
                    PrintWriter write = new PrintWriter(new BufferedWriter(new FileWriter(listfile)));
                    for (int i = 0; i < list.size(); i++) {
                        write.println((String) list.get(i));
                    }
                    write.close();
                } else {
                    listfile.delete();
                }
                out.println("The project was successfully deleted.");
            }
        } catch (Exception e) {
            out.println("Error accessing this project.");
        }
        out.println("<center><form><input type=button value=Continue onClick=\"opener.location.reload(); window.close()\"></form></center>");
        htmlFooter(out);
    }
```

**Fragmento 2 (`func2`):**
```java
public static void transfer(FileInputStream fileInStream, FileOutputStream fileOutStream) throws IOException {
        FileChannel fileInChannel = fileInStream.getChannel();
        FileChannel fileOutChannel = fileOutStream.getChannel();
        long fileInSize = fileInChannel.size();
        try {
            long transferred = fileInChannel.transferTo(0, fileInSize, fileOutChannel);
            if (transferred != fileInSize) {
                throw new IOException("transfer() did not complete");
            }
        } finally {
            ensureClose(fileInChannel, fileOutChannel);
        }
    }
```

---

