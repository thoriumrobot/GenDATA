/*
 * Copyright (c) 2014, 2017, Oracle and/or its affiliates. All rights reserved.
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
package jdk.jshell;

import org.checkerframework.dataflow.qual.Pure;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.TreeMap;
import javax.tools.JavaFileObject.Kind;
import static javax.tools.StandardLocation.CLASS_PATH;
import javax.tools.FileObject;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardLocation;
import static jdk.internal.jshell.debug.InternalDebugControl.DBG_FMGR;

class MemoryFileManager implements JavaFileManager {

    Iterable<? extends Path> getLocationAsPaths(Location loc);

    static abstract class MemoryJavaFileObject extends SimpleJavaFileObject {

        public MemoryJavaFileObject(String name, JavaFileObject.Kind kind) {
        }
    }

    class SourceMemoryJavaFileObject extends MemoryJavaFileObject {

        public Object getOrigin();

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors);
    }

    static class OutputMemoryJavaFileObject extends MemoryJavaFileObject {

        public OutputMemoryJavaFileObject(String name, JavaFileObject.Kind kind) {
        }

        public byte[] getBytes();

        public void dump();

        @Override
        public String getName();

        @Override
        public OutputStream openOutputStream() throws IOException;

        @Override
        public InputStream openInputStream() throws IOException;
    }

    public MemoryFileManager(StandardJavaFileManager standardManager, JShell proc) {
    }

    public void dumpClasses();

    public JavaFileObject createSourceFileObject(Object origin, String name, String code);

    @Override
    public ClassLoader getClassLoader(JavaFileManager.Location location);

    @Override
    public Iterable<JavaFileObject> list(JavaFileManager.Location location, String packageName, Set<JavaFileObject.Kind> kinds, boolean recurse) throws IOException;

    @Override
    public String inferBinaryName(JavaFileManager.Location location, JavaFileObject file);

    @Override
    public boolean isSameFile(FileObject a, FileObject b);

    @Override
    public int isSupportedOption(String option);

    @Override
    public boolean handleOption(String current, Iterator<String> remaining);

    @Override
    public boolean hasLocation(JavaFileManager.Location location);

    interface ClassFileCreationListener {

        void newClassFile(OutputMemoryJavaFileObject jfo, JavaFileManager.Location location, String className, Kind kind, FileObject sibling);
    }

    void registerClassFileCreationListener(ClassFileCreationListener listen);

    @Override
    public JavaFileObject getJavaFileForInput(JavaFileManager.Location location, String className, JavaFileObject.Kind kind) throws IOException;

    @Override
    public JavaFileObject getJavaFileForOutput(JavaFileManager.Location location, String className, Kind kind, FileObject sibling) throws IOException;

    @Override
    public FileObject getFileForInput(JavaFileManager.Location location, String packageName, String relativeName) throws IOException;

    @Override
    public FileObject getFileForOutput(JavaFileManager.Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

    @Override
    public Location getLocationForModule(Location location, String moduleName) throws IOException;

    @Override
    public Location getLocationForModule(Location location, JavaFileObject fo) throws IOException;

    @Override
    public String inferModuleName(Location location) throws IOException;

    @Override
    public Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

    @Override
    @Pure
    public boolean contains(Location location, FileObject file) throws IOException;

    @Override
    public void flush() throws IOException;

    @Override
    public void close() throws IOException;
}
