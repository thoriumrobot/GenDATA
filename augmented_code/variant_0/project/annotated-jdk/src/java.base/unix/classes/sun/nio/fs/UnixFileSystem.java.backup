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

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import java.nio.file.*;
import java.nio.file.attribute.*;
import java.nio.file.spi.*;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;
import sun.security.action.GetPropertyAction;

abstract class UnixFileSystem extends FileSystem {

    byte[] defaultDirectory();

    boolean needToResolveAgainstDefaultDirectory();

    UnixPath rootDirectory();

    static List<String> standardFileAttributeViews();

    @Override
    public final FileSystemProvider provider();

    @Override
    public final String getSeparator();

    @Override
    public final boolean isOpen();

    @Override
    public final boolean isReadOnly();

    @Override
    public final void close() throws IOException;

    void copyNonPosixAttributes(int sfd, int tfd);

    @Override
    public final Iterable<Path> getRootDirectories();

    abstract Iterable<UnixMountEntry> getMountEntries();

    abstract FileStore getFileStore(UnixMountEntry entry) throws IOException;

    private class FileStoreIterator implements Iterator<FileStore> {

        @Override
        @Pure
        public synchronized boolean hasNext();

        @Override
        @SideEffectsOnly("this")
        public synchronized FileStore next();

        @Override
        public void remove();
    }

    @Override
    public final Iterable<FileStore> getFileStores();

    @Override
    public final Path getPath(String first, String... more);

    @Override
    public PathMatcher getPathMatcher(String syntaxAndInput);

    @Override
    public final UserPrincipalLookupService getUserPrincipalLookupService();

    private static class LookupService {
    }

    Pattern compilePathMatchPattern(String expr);

    String normalizeNativePath(String path);

    String normalizeJavaPath(String path);
}
