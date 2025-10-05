/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2007, 2018, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.net.URI;
    @Positive
import java.nio.file.spi.FileSystemProvider;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.NoSuchElementException;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public interface Path extends Comparable<Path>, Iterable<Path>, Watchable {

    @Positive
    @SideEffectFree
    @Positive
    public static Path of(String first, String... more);

    @Positive
    @SideEffectFree
    @Positive
    public static Path of(URI uri);

    @Positive
    @Pure
    @Positive
    FileSystem getFileSystem();

    @Positive
    @Pure
    @Positive
    boolean isAbsolute();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    Path getRoot();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    Path getFileName();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    Path getParent();

    @Positive
    @Pure
    @Positive
    int getNameCount();

    @Positive
    @Pure
    @Positive
    Path getName(int index);

    @Positive
    @SideEffectFree
    @Positive
    Path subpath(int beginIndex, int endIndex);

    @Positive
    @Pure
    @Positive
    boolean startsWith(Path other);

    @Positive
    @Pure
    @Positive
    default boolean startsWith(String other);

    @Positive
    @Pure
    @Positive
    boolean endsWith(Path other);

    @Positive
    @Pure
    @Positive
    default boolean endsWith(String other);

    @Positive
    @SideEffectFree
    @Positive
    Path normalize();

    @Positive
    @SideEffectFree
    @Positive
    Path resolve(Path other);

    @Positive
    @SideEffectFree
    @Positive
    default Path resolve(String other);

    @Positive
    @SideEffectFree
    @Positive
    default Path resolveSibling(Path other);

    @Positive
    @SideEffectFree
    @Positive
    default Path resolveSibling(String other);

    @Positive
    @SideEffectFree
    @Positive
    Path relativize(Path other);

    @Positive
    @SideEffectFree
    @Positive
    URI toUri();

    @Positive
    @SideEffectFree
    @Positive
    Path toAbsolutePath();

    @Positive
    @SideEffectFree
    @Positive
    Path toRealPath(LinkOption... options) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    default File toFile();

    @Positive
    @Override
    @Positive
    WatchKey register(WatchService watcher, WatchEvent.Kind<?>[] events, WatchEvent.Modifier... modifiers) throws IOException;

    @Positive
    @Override
    @Positive
    default WatchKey register(WatchService watcher, WatchEvent.Kind<?>... events) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    default Iterator<Path> iterator();

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    int compareTo(Path other);

    @Positive
    @Pure
    @Positive
    boolean equals(@Nullable Object other);

    @Positive
    @Pure
    @Positive
    int hashCode();

    @Positive
    @SideEffectFree
    @Positive
    String toString();
    @Positive
}
