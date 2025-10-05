/*
 * Copyright (c) 2017, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.main;

import org.checkerframework.dataflow.qual.Pure;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Iterator;
import java.util.ServiceLoader;
import java.util.Set;
import javax.tools.FileObject;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileManager.Location;
import javax.tools.JavaFileObject;
import javax.tools.JavaFileObject.Kind;
import javax.tools.StandardJavaFileManager;
import com.sun.tools.javac.util.Context;

public class DelegatingJavaFileManager implements JavaFileManager {

    public static void installReleaseFileManager(Context context, JavaFileManager releaseFM, JavaFileManager originalFM);

    @Override
    public ClassLoader getClassLoader(Location location);

    @Override
    public Iterable<JavaFileObject> list(Location location, String packageName, Set<Kind> kinds, boolean recurse) throws IOException;

    @Override
    public String inferBinaryName(Location location, JavaFileObject file);

    @Override
    public boolean isSameFile(FileObject a, FileObject b);

    @Override
    public boolean handleOption(String current, Iterator<String> remaining);

    @Override
    public boolean hasLocation(Location location);

    @Override
    public JavaFileObject getJavaFileForInput(Location location, String className, Kind kind) throws IOException;

    @Override
    public JavaFileObject getJavaFileForOutput(Location location, String className, Kind kind, FileObject sibling) throws IOException;

    @Override
    public FileObject getFileForInput(Location location, String packageName, String relativeName) throws IOException;

    @Override
    public FileObject getFileForOutput(Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

    @Override
    public void flush() throws IOException;

    @Override
    public void close() throws IOException;

    @Override
    public Location getLocationForModule(Location location, String moduleName) throws IOException;

    @Override
    public Location getLocationForModule(Location location, JavaFileObject fo) throws IOException;

    @Override
    public <S> ServiceLoader<S> getServiceLoader(Location location, Class<S> service) throws IOException;

    @Override
    public String inferModuleName(Location location) throws IOException;

    @Override
    public Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

    @Override
    @Pure
    public boolean contains(Location location, FileObject fo) throws IOException;

    @Override
    public int isSupportedOption(String option);

    public JavaFileManager getBaseFileManager();

    private static final class DelegatingSJFM extends DelegatingJavaFileManager implements StandardJavaFileManager {

        @Override
        public boolean isSameFile(FileObject a, FileObject b);

        @Override
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromFiles(Iterable<? extends File> files);

        @Override
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromPaths(Collection<? extends Path> paths);

        @Deprecated()
        @Override
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromPaths(Iterable<? extends Path> paths);

        @Override
        public Iterable<? extends JavaFileObject> getJavaFileObjects(File... files);

        @Override
        public Iterable<? extends JavaFileObject> getJavaFileObjects(Path... paths);

        @Override
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromStrings(Iterable<String> names);

        @Override
        public Iterable<? extends JavaFileObject> getJavaFileObjects(String... names);

        @Override
        public void setLocation(Location location, Iterable<? extends File> files) throws IOException;

        @Override
        public void setLocationFromPaths(Location location, Collection<? extends Path> paths) throws IOException;

        @Override
        public void setLocationForModule(Location location, String moduleName, Collection<? extends Path> paths) throws IOException;

        @Override
        public Iterable<? extends File> getLocation(Location location);

        @Override
        public Iterable<? extends Path> getLocationAsPaths(Location location);

        @Override
        public Path asPath(FileObject file);

        @Override
        public void setPathFactory(PathFactory f);
    }
}
