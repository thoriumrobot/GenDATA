/*
 * Copyright (c) 2007, 2017, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.file;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.mustcall.qual.InheritableMustCall;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.nio.file.attribute.*;
import java.nio.file.spi.FileSystemProvider;
import java.util.Set;
import java.io.Closeable;
import java.io.IOException;

@AnnotatedFor({ "interning", "mustcall" })
@InheritableMustCall({})
@UsesObjectEquals
public abstract class FileSystem implements Closeable {

    protected FileSystem() {
    }

    public abstract FileSystemProvider provider();

    @Override
    public abstract void close() throws IOException;

    public abstract boolean isOpen();

    public abstract boolean isReadOnly();

    public abstract String getSeparator();

    public abstract Iterable<Path> getRootDirectories();

    public abstract Iterable<FileStore> getFileStores();

    public abstract Set<String> supportedFileAttributeViews();

    public abstract Path getPath(String first, String... more);

    public abstract PathMatcher getPathMatcher(String syntaxAndPattern);

    public abstract UserPrincipalLookupService getUserPrincipalLookupService();

    public abstract WatchService newWatchService() throws IOException;
}
