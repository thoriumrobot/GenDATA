/*
    @Positive
 * Copyright (c) 2007, 2017, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.file;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.mustcall.qual.InheritableMustCall;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.nio.file.attribute.*;
    @Positive
import java.nio.file.spi.FileSystemProvider;
    @Positive
import java.util.Set;
    @Positive
import java.io.Closeable;
    @Positive
import java.io.IOException;

    @Positive
@AnnotatedFor({ "interning", "mustcall" })
    @Positive
@InheritableMustCall({})
    @Positive
@UsesObjectEquals
    @Positive
public abstract class FileSystem implements Closeable {

    @Positive
    protected FileSystem() {
    @Positive
    }

    @Positive
    public abstract FileSystemProvider provider();

    @Positive
    @Override
    @Positive
    public abstract void close() throws IOException;

    @Positive
    public abstract boolean isOpen();

    @Positive
    public abstract boolean isReadOnly();

    @Positive
    public abstract String getSeparator();

    @Positive
    public abstract Iterable<Path> getRootDirectories();

    @Positive
    public abstract Iterable<FileStore> getFileStores();

    @Positive
    public abstract Set<String> supportedFileAttributeViews();

    @Positive
    public abstract Path getPath(String first, String... more);

    @Positive
    public abstract PathMatcher getPathMatcher(String syntaxAndPattern);

    @Positive
    public abstract UserPrincipalLookupService getUserPrincipalLookupService();

    @Positive
    public abstract WatchService newWatchService() throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
