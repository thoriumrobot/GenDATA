/*
 * Copyright (c) 2007, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.file.spi;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.nio.channels.AsynchronousFileChannel;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.SeekableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.AccessDeniedException;
import java.nio.file.AccessMode;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.CopyOption;
import java.nio.file.DirectoryNotEmptyException;
import java.nio.file.DirectoryStream;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.FileStore;
import java.nio.file.FileSystem;
import java.nio.file.FileSystemAlreadyExistsException;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.LinkPermission;
import java.nio.file.NoSuchFileException;
import java.nio.file.NotDirectoryException;
import java.nio.file.NotLinkException;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.net.URI;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.FileAttributeView;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.security.AccessController;
import java.security.PrivilegedAction;
import sun.nio.ch.FileChannelImpl;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class FileSystemProvider {

    protected FileSystemProvider() {
    }

    public static List<FileSystemProvider> installedProviders();

    public abstract String getScheme();

    public abstract FileSystem newFileSystem(URI uri, Map<String, ?> env) throws IOException;

    public abstract FileSystem getFileSystem(URI uri);

    public abstract Path getPath(URI uri);

    public FileSystem newFileSystem(Path path, Map<String, ?> env) throws IOException;

    public InputStream newInputStream(Path path, OpenOption... options) throws IOException;

    public OutputStream newOutputStream(Path path, OpenOption... options) throws IOException;

    public FileChannel newFileChannel(Path path, Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException;

    public AsynchronousFileChannel newAsynchronousFileChannel(Path path, Set<? extends OpenOption> options, ExecutorService executor, FileAttribute<?>... attrs) throws IOException;

    public abstract SeekableByteChannel newByteChannel(Path path, Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException;

    public abstract DirectoryStream<Path> newDirectoryStream(Path dir, DirectoryStream.Filter<? super Path> filter) throws IOException;

    public abstract void createDirectory(Path dir, FileAttribute<?>... attrs) throws IOException;

    public void createSymbolicLink(Path link, Path target, FileAttribute<?>... attrs) throws IOException;

    public void createLink(Path link, Path existing) throws IOException;

    public abstract void delete(Path path) throws IOException;

    public boolean deleteIfExists(Path path) throws IOException;

    public Path readSymbolicLink(Path link) throws IOException;

    public abstract void copy(Path source, Path target, CopyOption... options) throws IOException;

    public abstract void move(Path source, Path target, CopyOption... options) throws IOException;

    public abstract boolean isSameFile(Path path, Path path2) throws IOException;

    public abstract boolean isHidden(Path path) throws IOException;

    public abstract FileStore getFileStore(Path path) throws IOException;

    public abstract void checkAccess(Path path, AccessMode... modes) throws IOException;

    public abstract <V extends FileAttributeView> V getFileAttributeView(Path path, Class<V> type, LinkOption... options);

    public abstract <A extends BasicFileAttributes> A readAttributes(Path path, Class<A> type, LinkOption... options) throws IOException;

    public abstract Map<String, Object> readAttributes(Path path, String attributes, LinkOption... options) throws IOException;

    public abstract void setAttribute(Path path, String attribute, Object value, LinkOption... options) throws IOException;
}
