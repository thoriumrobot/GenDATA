/*
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.file;

import org.checkerframework.dataflow.qual.Pure;
import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.module.Configuration;
import java.lang.module.ModuleFinder;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.FileVisitOption;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.ProviderNotFoundException;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.spi.FileSystemProvider;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.lang.model.SourceVersion;
import javax.tools.FileObject;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import com.sun.tools.javac.file.RelativePath.RelativeDirectory;
import com.sun.tools.javac.file.RelativePath.RelativeFile;
import com.sun.tools.javac.main.Option;
import com.sun.tools.javac.resources.CompilerProperties.Errors;
import com.sun.tools.javac.util.Assert;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.Context.Factory;
import com.sun.tools.javac.util.DefinedBy;
import com.sun.tools.javac.util.DefinedBy.Api;
import com.sun.tools.javac.util.List;
import com.sun.tools.javac.util.ListBuffer;
import static java.nio.file.FileVisitOption.FOLLOW_LINKS;
import static javax.tools.StandardLocation.*;

public class JavacFileManager extends BaseFileManager implements StandardJavaFileManager {

    public static char[] toArray(CharBuffer buffer);

    protected boolean symbolFileEnabled;

    protected enum SortFiles implements Comparator<Path> {

        FORWARD {

            @Override
            public int compare(Path f1, Path f2);
        }
        , REVERSE {

            @Override
            public int compare(Path f1, Path f2);
        }

    }

    protected SortFiles sortFiles;

    public static void preRegister(Context context);

    public JavacFileManager(Context context, boolean register, Charset charset) {
    }

    @Override
    public void setContext(Context context);

    @Override
    @DefinedBy(DefinedBy.Api.COMPILER)
    public void setPathFactory(PathFactory f);

    public void setSymbolFileEnabled(boolean b);

    public boolean isSymbolFileEnabled();

    public JavaFileObject getJavaFileObject(String name);

    public JavaFileObject getJavaFileObject(Path file);

    public JavaFileObject getFileForOutput(String classname, JavaFileObject.Kind kind, JavaFileObject sibling) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public Iterable<? extends JavaFileObject> getJavaFileObjectsFromStrings(Iterable<String> names);

    @Override
    @DefinedBy(Api.COMPILER)
    public Iterable<? extends JavaFileObject> getJavaFileObjects(String... names);

    public static void testName(String name, boolean isValidPackageName, boolean isValidClassName);

    synchronized Container getContainer(Path path) throws IOException;

    private interface Container {

        public abstract void list(Path userPath, RelativeDirectory subdirectory, Set<JavaFileObject.Kind> fileKinds, boolean recurse, ListBuffer<JavaFileObject> resultList) throws IOException;

        public abstract JavaFileObject getFileObject(Path userPath, RelativeFile name) throws IOException;

        public abstract void close() throws IOException;

        public abstract boolean maintainsDirectoryIndex();

        public abstract Iterable<RelativeDirectory> indexedDirectories();
    }

    private final class JRTImageContainer implements Container {

        @Override
        public void list(Path userPath, RelativeDirectory subdirectory, Set<JavaFileObject.Kind> fileKinds, boolean recurse, ListBuffer<JavaFileObject> resultList) throws IOException;

        @Override
        public JavaFileObject getFileObject(Path userPath, RelativeFile name) throws IOException;

        @Override
        public void close() throws IOException;

        @Override
        public boolean maintainsDirectoryIndex();

        @Override
        public Iterable<RelativeDirectory> indexedDirectories();
    }

    private final class DirectoryContainer implements Container {

        public DirectoryContainer(Path directory) {
        }

        @Override
        public void list(Path userPath, RelativeDirectory subdirectory, Set<JavaFileObject.Kind> fileKinds, boolean recurse, ListBuffer<JavaFileObject> resultList) throws IOException;

        @Override
        public JavaFileObject getFileObject(Path userPath, RelativeFile name) throws IOException;

        @Override
        public void close() throws IOException;

        @Override
        public boolean maintainsDirectoryIndex();

        @Override
        public Iterable<RelativeDirectory> indexedDirectories();
    }

    private final class ArchiveContainer implements Container {

        public ArchiveContainer(Path archivePath) throws IOException, ProviderNotFoundException, SecurityException {
        }

        @Override
        public void list(Path userPath, RelativeDirectory subdirectory, Set<JavaFileObject.Kind> fileKinds, boolean recurse, ListBuffer<JavaFileObject> resultList) throws IOException;

        @Override
        public JavaFileObject getFileObject(Path userPath, RelativeFile name) throws IOException;

        @Override
        public void close() throws IOException;

        @Override
        public boolean maintainsDirectoryIndex();

        @Override
        public Iterable<RelativeDirectory> indexedDirectories();
    }

    @Override
    @DefinedBy(Api.COMPILER)
    public void flush();

    @Override
    @DefinedBy(Api.COMPILER)
    public void close() throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public ClassLoader getClassLoader(Location location);

    @Override
    @DefinedBy(Api.COMPILER)
    public Iterable<JavaFileObject> list(Location location, String packageName, Set<JavaFileObject.Kind> kinds, boolean recurse) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public String inferBinaryName(Location location, JavaFileObject file);

    @Override
    @DefinedBy(Api.COMPILER)
    public boolean isSameFile(FileObject a, FileObject b);

    @Override
    @DefinedBy(Api.COMPILER)
    public boolean hasLocation(Location location);

    protected boolean hasExplicitLocation(Location location);

    @Override
    @DefinedBy(Api.COMPILER)
    public JavaFileObject getJavaFileForInput(Location location, String className, JavaFileObject.Kind kind) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public FileObject getFileForInput(Location location, String packageName, String relativeName) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public JavaFileObject getJavaFileForOutput(Location location, String className, JavaFileObject.Kind kind, FileObject sibling) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public FileObject getFileForOutput(Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public Iterable<? extends JavaFileObject> getJavaFileObjectsFromFiles(Iterable<? extends File> files);

    @Override
    @DefinedBy(Api.COMPILER)
    public Iterable<? extends JavaFileObject> getJavaFileObjectsFromPaths(Collection<? extends Path> paths);

    @Override
    @DefinedBy(Api.COMPILER)
    public Iterable<? extends JavaFileObject> getJavaFileObjects(File... files);

    @Override
    @DefinedBy(Api.COMPILER)
    public Iterable<? extends JavaFileObject> getJavaFileObjects(Path... paths);

    @Override
    @DefinedBy(Api.COMPILER)
    public void setLocation(Location location, Iterable<? extends File> searchpath) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public void setLocationFromPaths(Location location, Collection<? extends Path> searchpath) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public Iterable<? extends File> getLocation(Location location);

    @Override
    @DefinedBy(Api.COMPILER)
    public Collection<? extends Path> getLocationAsPaths(Location location);

    private static class PathAndContainer implements Comparable<PathAndContainer> {

        @Override
        public int compareTo(PathAndContainer other);

        @Override
        public boolean equals(Object o);

        @Override
        public int hashCode();
    }

    @Override
    @DefinedBy(Api.COMPILER)
    @Pure
    public boolean contains(Location location, FileObject fo) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public Location getLocationForModule(Location location, String moduleName) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public <S> ServiceLoader<S> getServiceLoader(Location location, Class<S> service) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public Location getLocationForModule(Location location, JavaFileObject fo) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public void setLocationForModule(Location location, String moduleName, Collection<? extends Path> paths) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public String inferModuleName(Location location);

    @Override
    @DefinedBy(Api.COMPILER)
    public Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public Path asPath(FileObject file);

    protected static boolean isRelativeUri(URI uri);

    protected static boolean isRelativeUri(String u);

    public static String getRelativeName(File file);

    public static String getMessage(IOException e);

    @Override
    public boolean handleOption(Option option, String value);
}
