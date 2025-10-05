/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2011, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.api;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.Reader;
    @Positive
import java.io.Writer;
    @Positive
import java.lang.annotation.ElementType;
    @Positive
import java.lang.annotation.Retention;
    @Positive
import java.lang.annotation.RetentionPolicy;
    @Positive
import java.lang.annotation.Target;
    @Positive
import java.net.URI;
    @Positive
import java.nio.file.Path;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.Set;
    @Positive
import javax.lang.model.element.Modifier;
    @Positive
import javax.lang.model.element.NestingKind;
    @Positive
import javax.tools.Diagnostic;
    @Positive
import javax.tools.DiagnosticListener;
    @Positive
import javax.tools.FileObject;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileManager.Location;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import javax.tools.JavaFileObject.Kind;
    @Positive
import javax.tools.StandardJavaFileManager;
    @Positive
import com.sun.source.util.TaskEvent;
    @Positive
import com.sun.source.util.TaskListener;
    @Positive
import com.sun.tools.javac.util.ClientCodeException;
    @Positive
import com.sun.tools.javac.util.Context;
    @Positive
import com.sun.tools.javac.util.DefinedBy;
    @Positive
import com.sun.tools.javac.util.DefinedBy.Api;
    @Positive
import com.sun.tools.javac.util.JCDiagnostic;

    @Positive
public class ClientCodeWrapper {

    @Positive
    @Retention(RetentionPolicy.RUNTIME)
    @Positive
    @Target(ElementType.TYPE)
    @Positive
    public @interface Trusted {
    @Positive
    }

    @Positive
    public static ClientCodeWrapper instance(Context context);

    @Positive
    protected ClientCodeWrapper(Context context) {
    @Positive
    }

    @Positive
    public JavaFileManager wrap(JavaFileManager fm);

    @Positive
    public FileObject wrap(FileObject fo);

    @Positive
    FileObject unwrap(FileObject fo);

    @Positive
    public JavaFileObject wrap(JavaFileObject fo);

    @Positive
    public Iterable<JavaFileObject> wrapJavaFileObjects(Iterable<? extends JavaFileObject> list);

    @Positive
    JavaFileObject unwrap(JavaFileObject fo);

    @Positive
    public <T> DiagnosticListener<T> wrap(DiagnosticListener<T> dl);

    @Positive
    TaskListener wrap(TaskListener tl);

    @Positive
    TaskListener unwrap(TaskListener l);

    @Positive
    Collection<TaskListener> unwrap(Collection<? extends TaskListener> listeners);

    @Positive
    protected boolean isTrusted(Object o);

    @Positive
    protected class WrappedJavaFileManager implements JavaFileManager {

    @Positive
        protected JavaFileManager clientJavaFileManager;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public ClassLoader getClassLoader(Location location);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Iterable<JavaFileObject> list(Location location, String packageName, Set<Kind> kinds, boolean recurse) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public String inferBinaryName(Location location, JavaFileObject file);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public boolean isSameFile(FileObject a, FileObject b);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public boolean handleOption(String current, Iterator<String> remaining);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public boolean hasLocation(Location location);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public JavaFileObject getJavaFileForInput(Location location, String className, Kind kind) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public JavaFileObject getJavaFileForOutput(Location location, String className, Kind kind, FileObject sibling) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public FileObject getFileForInput(Location location, String packageName, String relativeName) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public FileObject getFileForOutput(Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        @Pure
    @Positive
        public boolean contains(Location location, FileObject file) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public void flush() throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public void close() throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Location getLocationForModule(Location location, String moduleName) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Location getLocationForModule(Location location, JavaFileObject fo) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public String inferModuleName(Location location) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public int isSupportedOption(String option);

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public <S> ServiceLoader<S> getServiceLoader(Location location, Class<S> service) throws IOException;
    @Positive
    }

    @Positive
    protected class WrappedStandardJavaFileManager extends WrappedJavaFileManager implements StandardJavaFileManager {

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromFiles(Iterable<? extends File> files);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromPaths(Collection<? extends Path> paths);

    @Positive
        @Deprecated()
    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromPaths(Iterable<? extends Path> paths);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Iterable<? extends JavaFileObject> getJavaFileObjects(File... files);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Iterable<? extends JavaFileObject> getJavaFileObjects(Path... paths);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Iterable<? extends JavaFileObject> getJavaFileObjectsFromStrings(Iterable<String> names);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Iterable<? extends JavaFileObject> getJavaFileObjects(String... names);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public void setLocation(Location location, Iterable<? extends File> files) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public void setLocationFromPaths(Location location, Collection<? extends Path> paths) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Iterable<? extends File> getLocation(Location location);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Iterable<? extends Path> getLocationAsPaths(Location location);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Path asPath(FileObject file);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public void setPathFactory(PathFactory f);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public void setLocationForModule(Location location, String moduleName, Collection<? extends Path> paths) throws IOException;
    @Positive
    }

    @Positive
    protected class WrappedFileObject implements FileObject {

    @Positive
        protected FileObject clientFileObject;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public URI toUri();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public String getName();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public InputStream openInputStream() throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public OutputStream openOutputStream() throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Reader openReader(boolean ignoreEncodingErrors) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public CharSequence getCharContent(boolean ignoreEncodingErrors) throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Writer openWriter() throws IOException;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public long getLastModified();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public boolean delete();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    protected class WrappedJavaFileObject extends WrappedFileObject implements JavaFileObject {

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Kind getKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public boolean isNameCompatible(String simpleName, Kind kind);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public NestingKind getNestingKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Modifier getAccessLevel();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    protected class WrappedDiagnosticListener<T> implements DiagnosticListener<T> {

    @Positive
        protected DiagnosticListener<T> clientDiagnosticListener;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public void report(Diagnostic<? extends T> diagnostic);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public class DiagnosticSourceUnwrapper implements Diagnostic<JavaFileObject> {

    @Positive
        public final JCDiagnostic d;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Diagnostic.Kind getKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public JavaFileObject getSource();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public long getPosition();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public long getStartPosition();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public long getEndPosition();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public long getLineNumber();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public long getColumnNumber();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public String getCode();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public String getMessage(Locale locale);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    protected class WrappedTaskListener implements TaskListener {

    @Positive
        protected TaskListener clientTaskListener;

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER_TREE)
    @Positive
        public void started(TaskEvent ev);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER_TREE)
    @Positive
        public void finished(TaskEvent ev);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }
    @Positive
}
