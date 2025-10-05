/*
 * Copyright (c) 2011, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.api;

import org.checkerframework.dataflow.qual.Pure;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Reader;
import java.io.Writer;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.net.URI;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.Set;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.NestingKind;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticListener;
import javax.tools.FileObject;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileManager.Location;
import javax.tools.JavaFileObject;
import javax.tools.JavaFileObject.Kind;
import javax.tools.StandardJavaFileManager;
import com.sun.source.util.TaskEvent;
import com.sun.source.util.TaskListener;
import com.sun.tools.javac.util.ClientCodeException;
import com.sun.tools.javac.util.Context;
import com.sun.tools.javac.util.DefinedBy;
import com.sun.tools.javac.util.DefinedBy.Api;
import com.sun.tools.javac.util.JCDiagnostic;

public class ClientCodeWrapper {

    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.TYPE)
    public @interface Trusted {
    }

    public static ClientCodeWrapper instance(Context context);

    protected ClientCodeWrapper(Context context) {
    }

    public JavaFileManager wrap(JavaFileManager fm);

    public FileObject wrap(FileObject fo);

    FileObject unwrap(FileObject fo);

    public JavaFileObject wrap(JavaFileObject fo);

    public Iterable<JavaFileObject> wrapJavaFileObjects(Iterable<? extends JavaFileObject> list);

    JavaFileObject unwrap(JavaFileObject fo);

    public <T> DiagnosticListener<T> wrap(DiagnosticListener<T> dl);

    TaskListener wrap(TaskListener tl);

    TaskListener unwrap(TaskListener l);

    Collection<TaskListener> unwrap(Collection<? extends TaskListener> listeners);

    protected boolean isTrusted(Object o);

    protected class WrappedJavaFileManager implements JavaFileManager {

        protected JavaFileManager clientJavaFileManager;

        @Override
        @DefinedBy(Api.COMPILER)
        public ClassLoader getClassLoader(Location location);

        @Override
        @DefinedBy(Api.COMPILER)
        public Iterable<JavaFileObject> list(Location location, String packageName, Set<Kind> kinds, boolean recurse) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public String inferBinaryName(Location location, JavaFileObject file);

        @Override
        @DefinedBy(Api.COMPILER)
        public boolean isSameFile(FileObject a, FileObject b);

        @Override
        @DefinedBy(Api.COMPILER)
        public boolean handleOption(String current, Iterator<String> remaining);

        @Override
        @DefinedBy(Api.COMPILER)
        public boolean hasLocation(Location location);

        @Override
        @DefinedBy(Api.COMPILER)
        public JavaFileObject getJavaFileForInput(Location location, String className, Kind kind) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public JavaFileObject getJavaFileForOutput(Location location, String className, Kind kind, FileObject sibling) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public FileObject getFileForInput(Location location, String packageName, String relativeName) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public FileObject getFileForOutput(Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        @Pure
        public boolean contains(Location location, FileObject file) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public void flush() throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public void close() throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public Location getLocationForModule(Location location, String moduleName) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public Location getLocationForModule(Location location, JavaFileObject fo) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public String inferModuleName(Location location) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public int isSupportedOption(String option);

        @Override
        public String toString();

        @Override
        @DefinedBy(Api.COMPILER)
        public <S> ServiceLoader<S> getServiceLoader(Location location, Class<S> service) throws IOException;
    }

    protected class WrappedStandardJavaFileManager extends WrappedJavaFileManager implements StandardJavaFileManager {

        @Override
        @DefinedBy(Api.COMPILER)
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromFiles(Iterable<? extends File> files);

        @Override
        @DefinedBy(Api.COMPILER)
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromPaths(Collection<? extends Path> paths);

        @Deprecated()
        @Override
        @DefinedBy(Api.COMPILER)
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromPaths(Iterable<? extends Path> paths);

        @Override
        @DefinedBy(Api.COMPILER)
        public Iterable<? extends JavaFileObject> getJavaFileObjects(File... files);

        @Override
        @DefinedBy(Api.COMPILER)
        public Iterable<? extends JavaFileObject> getJavaFileObjects(Path... paths);

        @Override
        @DefinedBy(Api.COMPILER)
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromStrings(Iterable<String> names);

        @Override
        @DefinedBy(Api.COMPILER)
        public Iterable<? extends JavaFileObject> getJavaFileObjects(String... names);

        @Override
        @DefinedBy(Api.COMPILER)
        public void setLocation(Location location, Iterable<? extends File> files) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public void setLocationFromPaths(Location location, Collection<? extends Path> paths) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public Iterable<? extends File> getLocation(Location location);

        @Override
        @DefinedBy(Api.COMPILER)
        public Iterable<? extends Path> getLocationAsPaths(Location location);

        @Override
        @DefinedBy(Api.COMPILER)
        public Path asPath(FileObject file);

        @Override
        @DefinedBy(Api.COMPILER)
        public void setPathFactory(PathFactory f);

        @Override
        @DefinedBy(Api.COMPILER)
        public void setLocationForModule(Location location, String moduleName, Collection<? extends Path> paths) throws IOException;
    }

    protected class WrappedFileObject implements FileObject {

        protected FileObject clientFileObject;

        @Override
        @DefinedBy(Api.COMPILER)
        public URI toUri();

        @Override
        @DefinedBy(Api.COMPILER)
        public String getName();

        @Override
        @DefinedBy(Api.COMPILER)
        public InputStream openInputStream() throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public OutputStream openOutputStream() throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public Reader openReader(boolean ignoreEncodingErrors) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public CharSequence getCharContent(boolean ignoreEncodingErrors) throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public Writer openWriter() throws IOException;

        @Override
        @DefinedBy(Api.COMPILER)
        public long getLastModified();

        @Override
        @DefinedBy(Api.COMPILER)
        public boolean delete();

        @Override
        public String toString();
    }

    protected class WrappedJavaFileObject extends WrappedFileObject implements JavaFileObject {

        @Override
        @DefinedBy(Api.COMPILER)
        public Kind getKind();

        @Override
        @DefinedBy(Api.COMPILER)
        public boolean isNameCompatible(String simpleName, Kind kind);

        @Override
        @DefinedBy(Api.COMPILER)
        public NestingKind getNestingKind();

        @Override
        @DefinedBy(Api.COMPILER)
        public Modifier getAccessLevel();

        @Override
        public String toString();
    }

    protected class WrappedDiagnosticListener<T> implements DiagnosticListener<T> {

        protected DiagnosticListener<T> clientDiagnosticListener;

        @Override
        @DefinedBy(Api.COMPILER)
        public void report(Diagnostic<? extends T> diagnostic);

        @Override
        public String toString();
    }

    public class DiagnosticSourceUnwrapper implements Diagnostic<JavaFileObject> {

        public final JCDiagnostic d;

        @Override
        @DefinedBy(Api.COMPILER)
        public Diagnostic.Kind getKind();

        @Override
        @DefinedBy(Api.COMPILER)
        public JavaFileObject getSource();

        @Override
        @DefinedBy(Api.COMPILER)
        public long getPosition();

        @Override
        @DefinedBy(Api.COMPILER)
        public long getStartPosition();

        @Override
        @DefinedBy(Api.COMPILER)
        public long getEndPosition();

        @Override
        @DefinedBy(Api.COMPILER)
        public long getLineNumber();

        @Override
        @DefinedBy(Api.COMPILER)
        public long getColumnNumber();

        @Override
        @DefinedBy(Api.COMPILER)
        public String getCode();

        @Override
        @DefinedBy(Api.COMPILER)
        public String getMessage(Locale locale);

        @Override
        public String toString();
    }

    protected class WrappedTaskListener implements TaskListener {

        protected TaskListener clientTaskListener;

        @Override
        @DefinedBy(Api.COMPILER_TREE)
        public void started(TaskEvent ev);

        @Override
        @DefinedBy(Api.COMPILER_TREE)
        public void finished(TaskEvent ev);

        @Override
        public String toString();
    }
}
