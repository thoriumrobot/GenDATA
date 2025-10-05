/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
public class UnixPath {
/*
    @Positive
 * Copyright (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.nio.file.*;
    @Positive
import java.nio.charset.*;
    @Positive
import java.io.*;
    @Positive
import java.net.URI;
    @Positive
import java.util.*;
    @Positive
import jdk.internal.access.JavaLangAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import static sun.nio.fs.UnixNativeDispatcher.*;
    @Positive
import static sun.nio.fs.UnixConstants.*;

    @Positive
class UnixPath implements Path {

    @Positive
    static String normalizeAndCheck(String input);

    @Positive
    byte[] asByteArray();

    @Positive
    byte[] getByteArrayForSysCalls();

    @Positive
    String getPathForExceptionMessage();

    @Positive
    String getPathForPermissionCheck();

    @Positive
    static UnixPath toUnixPath(Path obj);

    @Positive
    boolean isEmpty();

    @Positive
    @Override
    @Positive
    public UnixFileSystem getFileSystem();

    @Positive
    @Override
    @Positive
    public UnixPath getRoot();

    @Positive
    @Override
    @Positive
    public UnixPath getFileName();

    @Positive
    @Override
    @Positive
    public UnixPath getParent();

    @Positive
    @Override
    @Positive
    public int getNameCount();

    @Positive
    @Override
    @Positive
    public UnixPath getName(int index);

    @Positive
    @Override
    @Positive
    public UnixPath subpath(int beginIndex, int endIndex);

    @Positive
    @Override
    @Positive
    public boolean isAbsolute();

    @Positive
    @Override
    @Positive
    public UnixPath resolve(Path obj);

    @Positive
    UnixPath resolve(byte[] other);

    @Positive
    @Override
    @Positive
    public UnixPath relativize(Path obj);

    @Positive
    @Override
    @Positive
    public UnixPath normalize();

    @Positive
    @Override
    @Positive
    public boolean startsWith(Path other);

    @Positive
    @Override
    @Positive
    public boolean endsWith(Path other);

    @Positive
    @Override
    @Positive
    public int compareTo(Path other);

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
    int openForAttributeAccess(boolean followLinks) throws UnixException;

    @Positive
    void checkRead();

    @Positive
    void checkWrite();

    @Positive
    void checkDelete();

    @Positive
    @Override
    @Positive
    public UnixPath toAbsolutePath();

    @Positive
    @Override
    @Positive
    public Path toRealPath(LinkOption... options) throws IOException;

    @Positive
    @Override
    @Positive
    public URI toUri();

    @Positive
    @Override
    @Positive
    public WatchKey register(WatchService watcher, WatchEvent.Kind<?>[] events, WatchEvent.Modifier... modifiers) throws IOException;
    @Positive
}

}