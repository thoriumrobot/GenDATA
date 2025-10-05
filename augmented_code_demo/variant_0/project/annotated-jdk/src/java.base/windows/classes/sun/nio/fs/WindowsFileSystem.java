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
import java.util.*;
    @Positive
import java.util.regex.Pattern;
    @Positive
import java.io.IOException;

    @Positive
class WindowsFileSystem extends FileSystem {

    @Positive
    String defaultDirectory();

    @Positive
    String defaultRoot();

    @Positive
    @Override
    @Positive
    public FileSystemProvider provider();

    @Positive
    @Override
    @Positive
    public String getSeparator();

    @Positive
    @Override
    @Positive
    public boolean isOpen();

    @Positive
    @Override
    @Positive
    public boolean isReadOnly();

    @Positive
    @Override
    @Positive
    public void close() throws IOException;

    @Positive
    @Override
    @Positive
    public Iterable<Path> getRootDirectories();

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
    public Iterable<FileStore> getFileStores();

    @Positive
    @Override
    @Positive
    public Set<String> supportedFileAttributeViews();

    @Positive
    @Override
    @Positive
    public final Path getPath(String first, String... more);

    @Positive
    @Override
    @Positive
    public UserPrincipalLookupService getUserPrincipalLookupService();

    @Positive
    private static class LookupService {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public PathMatcher getPathMatcher(String syntaxAndInput);

    @Positive
    @Override
    @Positive
    public WatchService newWatchService() throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
