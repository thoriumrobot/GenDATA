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
import java.nio.file.attribute.*;
import java.io.*;
import java.net.URI;
import java.util.*;
import java.lang.ref.WeakReference;
import static sun.nio.fs.WindowsNativeDispatcher.*;
import static sun.nio.fs.WindowsConstants.*;

class WindowsPath implements Path {

    static WindowsPath parse(WindowsFileSystem fs, String path);

    static WindowsPath createFromNormalizedPath(WindowsFileSystem fs, String path, BasicFileAttributes attrs);

    static WindowsPath createFromNormalizedPath(WindowsFileSystem fs, String path);

    private static class WindowsPathWithAttributes extends WindowsPath implements BasicFileAttributesHolder {

        @Override
        public BasicFileAttributes get();

        @Override
        public void invalidate();
    }

    String getPathForExceptionMessage();

    String getPathForPermissionCheck();

    String getPathForWin32Calls() throws WindowsException;

    static String addPrefixIfNeeded(String path);

    @Override
    public WindowsFileSystem getFileSystem();

    @Override
    public Path getFileName();

    @Override
    public WindowsPath getParent();

    @Override
    public WindowsPath getRoot();

    WindowsPathType type();

    boolean isUnc();

    boolean needsSlashWhenResolving();

    @Override
    public boolean isAbsolute();

    static WindowsPath toWindowsPath(Path path);

    @Override
    public WindowsPath relativize(Path obj);

    @Override
    public WindowsPath normalize();

    @Override
    public WindowsPath resolve(Path obj);

    @Override
    public int getNameCount();

    @Override
    public WindowsPath getName(int index);

    @Override
    public WindowsPath subpath(int beginIndex, int endIndex);

    @Override
    public boolean startsWith(Path obj);

    @Override
    public boolean endsWith(Path obj);

    @Override
    public int compareTo(Path obj);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    @Override
    public String toString();

    long openForReadAttributeAccess(boolean followLinks) throws WindowsException;

    void checkRead();

    void checkWrite();

    void checkDelete();

    @Override
    public URI toUri();

    @Override
    public WindowsPath toAbsolutePath();

    @Override
    public WindowsPath toRealPath(LinkOption... options) throws IOException;

    @Override
    public WatchKey register(WatchService watcher, WatchEvent.Kind<?>[] events, WatchEvent.Modifier... modifiers) throws IOException;
}
