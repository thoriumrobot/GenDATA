/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package java.util.jar;

    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import java.io.*;
    @Positive
import java.net.URL;
    @Positive
import java.util.*;
    @Positive
import java.security.*;
    @Positive
import java.security.cert.CertificateException;
    @Positive
import java.util.zip.ZipEntry;
    @Positive
import jdk.internal.util.jar.JarIndex;
    @Positive
import sun.security.util.ManifestDigester;
    @Positive
import sun.security.util.ManifestEntryVerifier;
    @Positive
import sun.security.util.SignatureFileVerifier;
    @Positive
import sun.security.util.Debug;

    @Positive
class JarVerifier {

    @Positive
    public JarVerifier(byte[] rawBytes) {
    @Positive
    }

    @Positive
    public void beginEntry(JarEntry je, ManifestEntryVerifier mev) throws IOException;

    @Positive
    public void update(int b, ManifestEntryVerifier mev) throws IOException;

    @Positive
    public void update(int n, byte[] b, int off, int len, ManifestEntryVerifier mev) throws IOException;

    @Positive
    @Deprecated
    @Positive
    public java.security.cert.Certificate[] getCerts(String name);

    @Positive
    public java.security.cert.Certificate[] getCerts(JarFile jar, JarEntry entry);

    @Positive
    public CodeSigner[] getCodeSigners(String name);

    @Positive
    public CodeSigner[] getCodeSigners(JarFile jar, JarEntry entry);

    @Positive
    boolean nothingToVerify();

    @Positive
    void doneWithMeta();

    @Positive
    static class VerifierStream extends java.io.InputStream {

    @Positive
        public int read() throws IOException;

    @Positive
        public int read(byte[] b, int off, int len) throws IOException;

    @Positive
        public void close() throws IOException;

    @Positive
        public int available() throws IOException;
    @Positive
    }

    @Positive
    private static class VerifierCodeSource extends CodeSource {

    @Positive
        public boolean equals(Object obj);

    @Positive
        boolean isSameDomain(Object csdomain);
    @Positive
    }

    @Positive
    public synchronized Enumeration<String> entryNames(JarFile jar, final CodeSource[] cs);

    @Positive
    public Enumeration<JarEntry> entries2(final JarFile jar, Enumeration<JarEntry> e);

    @Positive
    static boolean isSigningRelated(String name);

    @Positive
    public synchronized CodeSource[] getCodeSources(JarFile jar, URL url);

    @Positive
    public CodeSource getCodeSource(URL url, String name);

    @Positive
    public CodeSource getCodeSource(URL url, JarFile jar, JarEntry je);

    @Positive
    public void setEagerValidation(boolean eager);

    @Positive
    public synchronized List<Object> getManifestDigests();

    @Positive
    static CodeSource getUnsignedCS(URL url);

    @Positive
    boolean isTrustedManifestEntry(String name);
    @Positive
}
