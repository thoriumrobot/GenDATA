/*
 * Copyright (c) 2007, 2018, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.spi.FileSystemProvider;
import java.util.Iterator;
import java.util.NoSuchElementException;

@AnnotatedFor({ "nullness" })
public interface Path extends Comparable<Path>, Iterable<Path>, Watchable {

    @SideEffectFree
    public static Path of(String first, String... more);

    @SideEffectFree
    public static Path of(URI uri);

    @Pure
    FileSystem getFileSystem();

    @Pure
    boolean isAbsolute();

    @Pure
    @Nullable
    Path getRoot();

    @Pure
    @Nullable
    Path getFileName();

    @Pure
    @Nullable
    Path getParent();

    @Pure
    int getNameCount();

    @Pure
    Path getName(int index);

    @SideEffectFree
    Path subpath(int beginIndex, int endIndex);

    @Pure
    boolean startsWith(Path other);

    @Pure
    default boolean startsWith(String other);

    @Pure
    boolean endsWith(Path other);

    @Pure
    default boolean endsWith(String other);

    @SideEffectFree
    Path normalize();

    @SideEffectFree
    Path resolve(Path other);

    @SideEffectFree
    default Path resolve(String other);

    @SideEffectFree
    default Path resolveSibling(Path other);

    @SideEffectFree
    default Path resolveSibling(String other);

    @SideEffectFree
    Path relativize(Path other);

    @SideEffectFree
    URI toUri();

    @SideEffectFree
    Path toAbsolutePath();

    @SideEffectFree
    Path toRealPath(LinkOption... options) throws IOException;

    @SideEffectFree
    default File toFile();

    @Override
    WatchKey register(WatchService watcher, WatchEvent.Kind<?>[] events, WatchEvent.Modifier... modifiers) throws IOException;

    @Override
    default WatchKey register(WatchService watcher, WatchEvent.Kind<?>... events) throws IOException;

    @SideEffectFree
    @Override
    default Iterator<Path> iterator();

    @Pure
    @Override
    int compareTo(Path other);

    @Pure
    boolean equals(@Nullable Object other);

    @Pure
    int hashCode();

    @SideEffectFree
    String toString();
}
