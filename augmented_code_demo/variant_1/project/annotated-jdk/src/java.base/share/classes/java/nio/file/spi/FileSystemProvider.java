/*
    @Positive
 * Copyright (c) 2007, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.file.spi;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.nio.channels.AsynchronousFileChannel;
    @Positive
import java.nio.channels.Channels;
    @Positive
import java.nio.channels.FileChannel;
    @Positive
import java.nio.channels.ReadableByteChannel;
    @Positive
import java.nio.channels.SeekableByteChannel;
    @Positive
import java.nio.channels.WritableByteChannel;
    @Positive
import java.nio.file.AccessDeniedException;
    @Positive
import java.nio.file.AccessMode;
    @Positive
import java.nio.file.AtomicMoveNotSupportedException;
    @Positive
import java.nio.file.CopyOption;
    @Positive
import java.nio.file.DirectoryNotEmptyException;
    @Positive
import java.nio.file.DirectoryStream;
    @Positive
import java.nio.file.FileAlreadyExistsException;
    @Positive
import java.nio.file.FileStore;
    @Positive
import java.nio.file.FileSystem;
    @Positive
import java.nio.file.FileSystemAlreadyExistsException;
    @Positive
import java.nio.file.FileSystemNotFoundException;
    @Positive
import java.nio.file.FileSystems;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.LinkOption;
    @Positive
import java.nio.file.LinkPermission;
    @Positive
import java.nio.file.NoSuchFileException;
    @Positive
import java.nio.file.NotDirectoryException;
    @Positive
import java.nio.file.NotLinkException;
    @Positive
import java.nio.file.OpenOption;
    @Positive
import java.nio.file.Path;
    @Positive
import java.nio.file.StandardOpenOption;
    @Positive
import java.net.URI;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.nio.file.attribute.BasicFileAttributes;
    @Positive
import java.nio.file.attribute.FileAttribute;
    @Positive
import java.nio.file.attribute.FileAttributeView;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.ServiceConfigurationError;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.Set;
    @Positive
import java.util.concurrent.ExecutorService;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import sun.nio.ch.FileChannelImpl;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class FileSystemProvider {

    @Positive
    protected FileSystemProvider() {
    @Positive
    }

    @Positive
    public static List<FileSystemProvider> installedProviders();

    @Positive
    public abstract String getScheme();

    @Positive
    public abstract FileSystem newFileSystem(URI uri, Map<String, ?> env) throws IOException;

    @Positive
    public abstract FileSystem getFileSystem(URI uri);

    @Positive
    public abstract Path getPath(URI uri);

    @Positive
    public FileSystem newFileSystem(Path path, Map<String, ?> env) throws IOException;

    @Positive
    public InputStream newInputStream(Path path, OpenOption... options) throws IOException;

    @Positive
    public OutputStream newOutputStream(Path path, OpenOption... options) throws IOException;

    @Positive
    public FileChannel newFileChannel(Path path, Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException;

    @Positive
    public AsynchronousFileChannel newAsynchronousFileChannel(Path path, Set<? extends OpenOption> options, ExecutorService executor, FileAttribute<?>... attrs) throws IOException;

    @Positive
    public abstract SeekableByteChannel newByteChannel(Path path, Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException;

    @Positive
    public abstract DirectoryStream<Path> newDirectoryStream(Path dir, DirectoryStream.Filter<? super Path> filter) throws IOException;

    @Positive
    public abstract void createDirectory(Path dir, FileAttribute<?>... attrs) throws IOException;

    @Positive
    public void createSymbolicLink(Path link, Path target, FileAttribute<?>... attrs) throws IOException;

    @Positive
    public void createLink(Path link, Path existing) throws IOException;

    @Positive
    public abstract void delete(Path path) throws IOException;

    @Positive
    public boolean deleteIfExists(Path path) throws IOException;

    @Positive
    public Path readSymbolicLink(Path link) throws IOException;

    @Positive
    public abstract void copy(Path source, Path target, CopyOption... options) throws IOException;

    @Positive
    public abstract void move(Path source, Path target, CopyOption... options) throws IOException;

    @Positive
    public abstract boolean isSameFile(Path path, Path path2) throws IOException;

    @Positive
    public abstract boolean isHidden(Path path) throws IOException;

    @Positive
    public abstract FileStore getFileStore(Path path) throws IOException;

    @Positive
    public abstract void checkAccess(Path path, AccessMode... modes) throws IOException;

    @Positive
    public abstract <V extends FileAttributeView> V getFileAttributeView(Path path, Class<V> type, LinkOption... options);

    @Positive
    public abstract <A extends BasicFileAttributes> A readAttributes(Path path, Class<A> type, LinkOption... options) throws IOException;

    @Positive
    public abstract Map<String, Object> readAttributes(Path path, String attributes, LinkOption... options) throws IOException;

    @Positive
    public abstract void setAttribute(Path path, String attribute, Object value, LinkOption... options) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 1
