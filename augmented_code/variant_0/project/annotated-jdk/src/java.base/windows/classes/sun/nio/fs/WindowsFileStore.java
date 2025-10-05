/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class WindowsFileStore {
/*
    @Copyright * Positive (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.nio.fs;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.nio.file.FileStore;
    @Positive
import java.nio.file.FileSystemException;
    @Positive
import java.nio.file.attribute.AclFileAttributeView;
    @Positive
import java.nio.file.attribute.BasicFileAttributeView;
    @Positive
import java.nio.file.attribute.DosFileAttributeView;
    @Positive
import java.nio.file.attribute.FileAttributeView;
    @Positive
import java.nio.file.attribute.FileOwnerAttributeView;
    @Positive
import java.nio.file.attribute.FileStoreAttributeView;
    @Positive
import java.nio.file.attribute.UserDefinedFileAttributeView;
    @Positive
import java.io.IOException;
    @Positive
import java.util.Locale;
    @Positive
import static sun.nio.fs.WindowsConstants.*;
    @Positive
import static sun.nio.fs.WindowsNativeDispatcher.*;

    @Positive
class WindowsFileStore extends FileStore {

    @Positive
    static WindowsFileStore create(String root, boolean ignoreNotReady) throws IOException;

    @Positive
    static WindowsFileStore create(WindowsPath file) throws IOException;

    @Positive
    VolumeInformation volumeInformation();

    @Positive
    int volumeType();

    @Positive
    @Override
    @Positive
    public String name();

    @Positive
    @Override
    @Positive
    public String type();

    @Positive
    @Override
    @Positive
    public boolean isReadOnly();

    @Positive
    @Override
    @Positive
    public long getTotalSpace() throws IOException;

    @Positive
    @Override
    @Positive
    public long getUsableSpace() throws IOException;

    @Positive
    @Override
    @Positive
    public long getUnallocatedSpace() throws IOException;

    @Positive
    @Override
    @Positive
    public long getBlockSize() throws IOException;

    @Positive
    @Override
    @Positive
    public <V extends FileStoreAttributeView> V getFileStoreAttributeView(Class<V> type);

    @Positive
    @Override
    @Positive
    public Object getAttribute(String attribute) throws IOException;

    @Positive
    @Override
    @Positive
    public boolean supportsFileAttributeView(Class<? extends FileAttributeView> type);

    @Positive
    @Override
    @Positive
    public boolean supportsFileAttributeView(String name);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object ob);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toString();
    @Positive
}

}