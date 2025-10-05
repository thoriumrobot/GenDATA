/*
 * Copyright (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package sun.nio.fs;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.nio.file.*;
import java.nio.charset.*;
import java.io.*;
import java.net.URI;
import java.util.*;
import jdk.internal.access.JavaLangAccess;
import jdk.internal.access.SharedSecrets;
import static sun.nio.fs.UnixNativeDispatcher.*;
import static sun.nio.fs.UnixConstants.*;

class UnixPath implements Path {

    static String normalizeAndCheck(String input);

    byte[] asByteArray();

    byte[] getByteArrayForSysCalls();

    String getPathForExceptionMessage();

    String getPathForPermissionCheck();

    static UnixPath toUnixPath(Path obj);

    boolean isEmpty();

    @Override
    public UnixFileSystem getFileSystem();

    @Override
    public UnixPath getRoot();

    @Override
    public UnixPath getFileName();

    @Override
    public UnixPath getParent();

    @Override
    public int getNameCount();

    @Override
    public UnixPath getName(int index);

    @Override
    public UnixPath subpath(int beginIndex, int endIndex);

    @Override
    public boolean isAbsolute();

    @Override
    public UnixPath resolve(Path obj);

    UnixPath resolve(byte[] other);

    @Override
    public UnixPath relativize(Path obj);

    @Override
    public UnixPath normalize();

    @Override
    public boolean startsWith(Path other);

    @Override
    public boolean endsWith(Path other);

    @Override
    public int compareTo(Path other);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object ob);

    @Override
    public int hashCode();

    @Override
    public String toString();

    int openForAttributeAccess(boolean followLinks) throws UnixException;

    void checkRead();

    void checkWrite();

    void checkDelete();

    @Override
    public UnixPath toAbsolutePath();

    @Override
    public Path toRealPath(LinkOption... options) throws IOException;

    @Override
    public URI toUri();

    @Override
    public WatchKey register(WatchService watcher, WatchEvent.Kind<?>[] events, WatchEvent.Modifier... modifiers) throws IOException;
}
