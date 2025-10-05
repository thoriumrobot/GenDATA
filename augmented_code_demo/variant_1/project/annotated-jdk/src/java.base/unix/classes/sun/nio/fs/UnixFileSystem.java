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
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.nio.file.*;
    @Positive
import java.nio.file.attribute.*;
    @Positive
import java.nio.file.spi.*;
    @Positive
import java.io.IOException;
    @Positive
import java.util.*;
    @Positive
import java.util.regex.Pattern;
    @Positive
import sun.security.action.GetPropertyAction;

    @Positive
abstract class UnixFileSystem extends FileSystem {

    @Positive
    byte[] defaultDirectory();

    @Positive
    boolean needToResolveAgainstDefaultDirectory();

    @Positive
    UnixPath rootDirectory();

    @Positive
    static List<String> standardFileAttributeViews();

    @Positive
    @Override
    @Positive
    public final FileSystemProvider provider();

    @Positive
    @Override
    @Positive
    public final String getSeparator();

    @Positive
    @Override
    @Positive
    public final boolean isOpen();

    @Positive
    @Override
    @Positive
    public final boolean isReadOnly();

    @Positive
    @Override
    @Positive
    public final void close() throws IOException;

    @Positive
    void copyNonPosixAttributes(int sfd, int tfd);

    @Positive
    @Override
    @Positive
    public final Iterable<Path> getRootDirectories();

    @Positive
    abstract Iterable<UnixMountEntry> getMountEntries();

    @Positive
    abstract FileStore getFileStore(UnixMountEntry entry) throws IOException;

    @Positive
    private class FileStoreIterator implements Iterator<FileStore> {

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public synchronized boolean hasNext();

    @Positive
        @Override
    @Positive
        @SideEffectsOnly("this")
    @Positive
        public synchronized FileStore next();

    @Positive
        @Override
    @Positive
        public void remove();
    @Positive
    }

    @Positive
    @Override
    @Positive
    public final Iterable<FileStore> getFileStores();

    @Positive
    @Override
    @Positive
    public final Path getPath(String first, String... more);

    @Positive
    @Override
    @Positive
    public PathMatcher getPathMatcher(String syntaxAndInput);

    @Positive
    @Override
    @Positive
    public final UserPrincipalLookupService getUserPrincipalLookupService();

    @Positive
    private static class LookupService {
    @Positive
    }

    @Positive
    Pattern compilePathMatchPattern(String expr);

    @Positive
    String normalizeNativePath(String path);

    @Positive
    String normalizeJavaPath(String path);
    @Positive
}

// CFWR semantic augmentation - variant 1
