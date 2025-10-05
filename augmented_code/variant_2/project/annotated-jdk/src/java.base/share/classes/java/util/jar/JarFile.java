/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.util.jar;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.common.value.qual.StringVal;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.access.JavaUtilZipFileAccess;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.security.util.ManifestEntryVerifier;
    @Positive
import java.io.ByteArrayInputStream;
    @Positive
import java.io.EOFException;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.net.URL;
    @Positive
import java.security.CodeSigner;
    @Positive
import java.security.CodeSource;
    @Positive
import java.security.cert.Certificate;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.zip.ZipEntry;
    @Positive
import java.util.zip.ZipException;
    @Positive
import java.util.zip.ZipFile;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public class JarFile extends ZipFile {

    @Positive
    @Interned
    @Positive
    @StringVal("META-INF/MANIFEST.MF")
    @Positive
    public static final String MANIFEST_NAME;

    @Positive
    public static Runtime.Version baseVersion();

    @Positive
    public static Runtime.Version runtimeVersion();

    @Positive
    public JarFile(String name) throws IOException {
    @Positive
    }

    @Positive
    public JarFile(String name, boolean verify) throws IOException {
    @Positive
    }

    @Positive
    public JarFile(File file) throws IOException {
    @Positive
    }

    @Positive
    public JarFile(File file, boolean verify) throws IOException {
    @Positive
    }

    @Positive
    public JarFile(File file, boolean verify, int mode) throws IOException {
    @Positive
    }

    @Positive
    public JarFile(File file, boolean verify, int mode, Runtime.Version version) throws IOException {
    @Positive
    }

    @Positive
    public final Runtime.Version getVersion();

    @Positive
    public final boolean isMultiRelease();

    @Positive
    @Nullable
    @Positive
    public Manifest getManifest() throws IOException;

    @Positive
    @Nullable
    @Positive
    public JarEntry getJarEntry(String name);

    @Positive
    @Nullable
    @Positive
    public ZipEntry getEntry(String name);

    @Positive
    public Enumeration<JarEntry> entries();

    @Positive
    public Stream<JarEntry> stream();

    @Positive
    public Stream<JarEntry> versionedStream();

    @Positive
    JarEntry entryFor(String name);

    @Positive
    String getRealName(JarEntry entry);

    @Positive
    private class JarFileEntry extends JarEntry {

    @Positive
        @Override
    @Positive
        @Nullable
    @Positive
        public Attributes getAttributes() throws IOException;

    @Positive
        @Override
    @Positive
        public Certificate @Nullable [] getCertificates();

    @Positive
        @Override
    @Positive
        public CodeSigner @Nullable [] getCodeSigners();

    @Positive
        @Override
    @Positive
        public String getRealName();

    @Positive
        @Override
    @Positive
        public String getName();

    @Positive
        JarFileEntry realEntry();

    @Positive
        JarFileEntry withBasename(String name);
    @Positive
    }

    @Positive
    public synchronized InputStream getInputStream(ZipEntry ze) throws IOException;

    @Positive
    boolean hasClassPathAttribute() throws IOException;

    @Positive
    synchronized void ensureInitialization();

    @Positive
    static boolean isInitializing();

    @Positive
    JarEntry newEntry(JarEntry je);

    @Positive
    JarEntry newEntry(String name);

    @Positive
    Enumeration<String> entryNames(CodeSource[] cs);

    @Positive
    Enumeration<JarEntry> entries2();

    @Positive
    CodeSource @Nullable [] getCodeSources(URL url);

    @Positive
    CodeSource getCodeSource(URL url, String name);

    @Positive
    void setEagerValidation(boolean eager);

    @Positive
    List<Object> getManifestDigests();
    @Positive
}
