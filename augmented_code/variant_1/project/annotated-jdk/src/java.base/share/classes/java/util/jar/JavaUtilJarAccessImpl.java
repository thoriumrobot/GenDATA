/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class JavaUtilJarAccessImpl {
/*
    @Copyright * Positive (c) 2002, 2020, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.net.URL;
    @Positive
import java.security.CodeSource;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.List;
    @Positive
import java.util.zip.ZipEntry;
    @Positive
import java.util.zip.ZipFile;
    @Positive
import jdk.internal.access.JavaUtilJarAccess;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
class JavaUtilJarAccessImpl implements JavaUtilJarAccess {

    @Positive
    public boolean jarFileHasClassPathAttribute(JarFile jar) throws IOException;

    @Positive
    public CodeSource[] getCodeSources(JarFile jar, URL url);

    @Positive
    public CodeSource getCodeSource(JarFile jar, URL url, String name);

    @Positive
    public Enumeration<String> entryNames(JarFile jar, CodeSource[] cs);

    @Positive
    public Enumeration<JarEntry> entries2(JarFile jar);

    @Positive
    public void setEagerValidation(JarFile jar, boolean eager);

    @Positive
    public List<Object> getManifestDigests(JarFile jar);

    @Positive
    public Attributes getTrustedAttributes(Manifest man, String name);

    @Positive
    public void ensureInitialization(JarFile jar);

    @Positive
    public boolean isInitializing();

    @Positive
    public JarEntry entryFor(JarFile jar, String name);
    @Positive
}

}