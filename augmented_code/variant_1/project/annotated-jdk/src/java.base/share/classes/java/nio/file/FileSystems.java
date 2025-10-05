/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2007, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.file;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.nio.file.spi.FileSystemProvider;
    @Positive
import java.net.URI;
    @Positive
import java.io.IOException;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Map;
    @Positive
import java.util.ServiceConfigurationError;
    @Positive
import java.util.ServiceLoader;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import sun.nio.fs.DefaultFileSystemProvider;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public final class FileSystems {

    @Positive
    private static class DefaultFileSystemHolder {
    @Positive
    }

    @Positive
    public static FileSystem getDefault();

    @Positive
    public static FileSystem getFileSystem(URI uri);

    @Positive
    public static FileSystem newFileSystem(URI uri, Map<String, ?> env) throws IOException;

    @Positive
    public static FileSystem newFileSystem(URI uri, Map<String, ?> env, ClassLoader loader) throws IOException;

    @Positive
    public static FileSystem newFileSystem(Path path, ClassLoader loader) throws IOException;

    @Positive
    public static FileSystem newFileSystem(Path path, Map<String, ?> env) throws IOException;

    @Positive
    public static FileSystem newFileSystem(Path path) throws IOException;

    @Positive
    public static FileSystem newFileSystem(Path path, Map<String, ?> env, ClassLoader loader) throws IOException;
    @Positive
}
