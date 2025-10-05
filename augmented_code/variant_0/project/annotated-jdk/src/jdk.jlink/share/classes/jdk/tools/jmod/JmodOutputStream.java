/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class JmodOutputStream {
/*
    @Copyright * Positive (c) 2016, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.tools.jmod;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.BufferedOutputStream;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.Path;
    @Positive
import java.nio.file.Paths;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.zip.ZipEntry;
    @Positive
import java.util.zip.ZipOutputStream;
    @Positive
import jdk.internal.jmod.JmodFile;
    @Positive
import static jdk.internal.jmod.JmodFile.*;

    @Positive
class JmodOutputStream extends OutputStream implements AutoCloseable {

    @Positive
    static JmodOutputStream newOutputStream(Path file) throws IOException;

    @Positive
    public void writeEntry(InputStream in, Section section, String name) throws IOException;

    @Positive
    public void writeEntry(byte[] bytes, Section section, String path) throws IOException;

    @Positive
    public void writeEntry(InputStream in, Entry e) throws IOException;

    @Positive
    @Pure
    @Positive
    public boolean contains(Section section, String path);

    @Positive
    @Override
    @Positive
    public void write(int b) throws IOException;

    @Positive
    @Override
    @Positive
    public void close() throws IOException;
    @Positive
}

}