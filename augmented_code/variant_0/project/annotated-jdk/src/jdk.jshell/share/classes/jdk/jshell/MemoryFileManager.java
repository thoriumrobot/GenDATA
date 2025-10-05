/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2014, 2017, Oracle and/or its affiliates. All rights reserved.
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
package jdk.jshell;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.ByteArrayInputStream;
    @Positive
import java.io.ByteArrayOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.net.URI;
    @Positive
import java.nio.file.FileSystems;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.Path;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Map;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Set;
    @Positive
import java.util.TreeMap;
    @Positive
import javax.tools.JavaFileObject.Kind;
    @Positive
import static javax.tools.StandardLocation.CLASS_PATH;
    @Positive
import javax.tools.FileObject;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import javax.tools.SimpleJavaFileObject;
    @Positive
import javax.tools.StandardJavaFileManager;
    @Positive
import javax.tools.StandardLocation;
    @Positive
import static jdk.internal.jshell.debug.InternalDebugControl.DBG_FMGR;

    @Positive
class MemoryFileManager implements JavaFileManager {

    @Positive
    Iterable<? extends Path> getLocationAsPaths(Location loc);

    @Positive
    static abstract class MemoryJavaFileObject extends SimpleJavaFileObject {

    @Positive
        public MemoryJavaFileObject(String name, JavaFileObject.Kind kind) {
    @Positive
        }
    @Positive
    }

    @Positive
    class SourceMemoryJavaFileObject extends MemoryJavaFileObject {

    @Positive
        public Object getOrigin();

    @Positive
        @Override
    @Positive
        public CharSequence getCharContent(boolean ignoreEncodingErrors);
    @Positive
    }

    @Positive
    static class OutputMemoryJavaFileObject extends MemoryJavaFileObject {

    @Positive
        public OutputMemoryJavaFileObject(String name, JavaFileObject.Kind kind) {
    @Positive
        }

    @Positive
        public byte[] getBytes();

    @Positive
        public void dump();

    @Positive
        @Override
    @Positive
        public String getName();

    @Positive
        @Override
    @Positive
        public OutputStream openOutputStream() throws IOException;

    @Positive
        @Override
    @Positive
        public InputStream openInputStream() throws IOException;
    @Positive
    }

    @Positive
    public MemoryFileManager(StandardJavaFileManager standardManager, JShell proc) {
    @Positive
    }

    @Positive
    public void dumpClasses();

    @Positive
    public JavaFileObject createSourceFileObject(Object origin, String name, String code);

    @Positive
    @Override
    @Positive
    public ClassLoader getClassLoader(JavaFileManager.Location location);

    @Positive
    @Override
    @Positive
    public Iterable<JavaFileObject> list(JavaFileManager.Location location, String packageName, Set<JavaFileObject.Kind> kinds, boolean recurse) throws IOException;

    @Positive
    @Override
    @Positive
    public String inferBinaryName(JavaFileManager.Location location, JavaFileObject file);

    @Positive
    @Override
    @Positive
    public boolean isSameFile(FileObject a, FileObject b);

    @Positive
    @Override
    @Positive
    public int isSupportedOption(String option);

    @Positive
    @Override
    @Positive
    public boolean handleOption(String current, Iterator<String> remaining);

    @Positive
    @Override
    @Positive
    public boolean hasLocation(JavaFileManager.Location location);

    @Positive
    interface ClassFileCreationListener {

    @Positive
        void newClassFile(OutputMemoryJavaFileObject jfo, JavaFileManager.Location location, String className, Kind kind, FileObject sibling);
    @Positive
    }

    @Positive
    void registerClassFileCreationListener(ClassFileCreationListener listen);

    @Positive
    @Override
    @Positive
    public JavaFileObject getJavaFileForInput(JavaFileManager.Location location, String className, JavaFileObject.Kind kind) throws IOException;

    @Positive
    @Override
    @Positive
    public JavaFileObject getJavaFileForOutput(JavaFileManager.Location location, String className, Kind kind, FileObject sibling) throws IOException;

    @Positive
    @Override
    @Positive
    public FileObject getFileForInput(JavaFileManager.Location location, String packageName, String relativeName) throws IOException;

    @Positive
    @Override
    @Positive
    public FileObject getFileForOutput(JavaFileManager.Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

    @Positive
    @Override
    @Positive
    public Location getLocationForModule(Location location, String moduleName) throws IOException;

    @Positive
    @Override
    @Positive
    public Location getLocationForModule(Location location, JavaFileObject fo) throws IOException;

    @Positive
    @Override
    @Positive
    public String inferModuleName(Location location) throws IOException;

    @Positive
    @Override
    @Positive
    public Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean contains(Location location, FileObject file) throws IOException;

    @Positive
    @Override
    @Positive
    public void flush() throws IOException;

    @Positive
    @Override
    @Positive
    public void close() throws IOException;
    @Positive
}
