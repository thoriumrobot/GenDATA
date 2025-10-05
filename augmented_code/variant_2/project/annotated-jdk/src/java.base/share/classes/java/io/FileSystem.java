/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1998, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.annotation.Native;

    @Positive
@AnnotatedFor({ "index", "interning" })
    @Positive
@UsesObjectEquals
    @Positive
abstract class FileSystem {

    @Positive
    public abstract char getSeparator();

    @Positive
    public abstract char getPathSeparator();

    @Positive
    public abstract String normalize(String path);

    @Positive
    @IndexOrHigh({ "#1" })
    @Positive
    public abstract int prefixLength(String path);

    @Positive
    public abstract String resolve(String parent, String child);

    @Positive
    public abstract String getDefaultParent();

    @Positive
    public abstract String fromURIPath(String path);

    @Positive
    public abstract boolean isAbsolute(File f);

    @Positive
    public abstract String resolve(File f);

    @Positive
    public abstract String canonicalize(String path) throws IOException;

    @Positive
    @Native
    @Positive
    @SignedPositive
    @Positive
    public static final int BA_EXISTS;

    @Positive
    @Native
    @Positive
    @SignedPositive
    @Positive
    public static final int BA_REGULAR;

    @Positive
    @Native
    @Positive
    @SignedPositive
    @Positive
    public static final int BA_DIRECTORY;

    @Positive
    @Native
    @Positive
    @SignedPositive
    @Positive
    public static final int BA_HIDDEN;

    @Positive
    public abstract int getBooleanAttributes(File f);

    @Positive
    public boolean hasBooleanAttributes(File f, int attributes);

    @Positive
    @Native
    @Positive
    @SignedPositive
    @Positive
    public static final int ACCESS_READ;

    @Positive
    @Native
    @Positive
    @SignedPositive
    @Positive
    public static final int ACCESS_WRITE;

    @Positive
    @Native
    @Positive
    @SignedPositive
    @Positive
    public static final int ACCESS_EXECUTE;

    @Positive
    public abstract boolean checkAccess(File f, int access);

    @Positive
    public abstract boolean setPermission(File f, int access, boolean enable, boolean owneronly);

    @Positive
    public abstract long getLastModifiedTime(File f);

    @Positive
    public abstract long getLength(File f);

    @Positive
    public abstract boolean createFileExclusively(String pathname) throws IOException;

    @Positive
    public abstract boolean delete(File f);

    @Positive
    public abstract String[] list(File f);

    @Positive
    public abstract boolean createDirectory(File f);

    @Positive
    public abstract boolean rename(File f1, File f2);

    @Positive
    public abstract boolean setLastModifiedTime(File f, long time);

    @Positive
    public abstract boolean setReadOnly(File f);

    @Positive
    public abstract File[] listRoots();

    @Positive
    @Native
    @Positive
    public static final int SPACE_TOTAL;

    @Positive
    @Native
    @Positive
    public static final int SPACE_FREE;

    @Positive
    @Native
    @Positive
    public static final int SPACE_USABLE;

    @Positive
    public abstract long getSpace(File f, int t);

    @Positive
    public abstract int getNameMax(String path);

    @Positive
    public abstract int compare(File f1, File f2);

    @Positive
    public abstract int hashCode(File f);
    @Positive
}
