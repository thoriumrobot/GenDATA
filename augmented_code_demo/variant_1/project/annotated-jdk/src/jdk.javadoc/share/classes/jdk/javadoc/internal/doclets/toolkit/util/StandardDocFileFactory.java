/*
    @Positive
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.BufferedInputStream;
    @Positive
import java.io.BufferedOutputStream;
    @Positive
import java.io.BufferedWriter;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.OutputStreamWriter;
    @Positive
import java.io.UnsupportedEncodingException;
    @Positive
import java.io.Writer;
    @Positive
import java.nio.file.DirectoryStream;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.Path;
    @Positive
import java.nio.file.Paths;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Set;
    @Positive
import javax.tools.DocumentationTool;
    @Positive
import javax.tools.FileObject;
    @Positive
import javax.tools.JavaFileManager.Location;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import javax.tools.StandardJavaFileManager;
    @Positive
import javax.tools.StandardLocation;
    @Positive
import com.sun.tools.javac.util.Assert;
    @Positive
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;

    @Positive
class StandardDocFileFactory extends DocFileFactory {

    @Positive
    public StandardDocFileFactory(BaseConfiguration configuration) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void setDestDir(String destDirName) throws SimpleDocletException;

    @Positive
    @Override
    @Positive
    public DocFile createFileForDirectory(String file);

    @Positive
    @Override
    @Positive
    public DocFile createFileForInput(String file);

    @Positive
    @Override
    @Positive
    public DocFile createFileForInput(Path file);

    @Positive
    @Override
    @Positive
    public DocFile createFileForOutput(DocPath path);

    @Positive
    @Override
    @Positive
    Iterable<DocFile> list(Location location, DocPath path);

    @Positive
    class StandardDocFile extends DocFile {

    @Positive
        @Override
    @Positive
        public FileObject getFileObject();

    @Positive
        @Override
    @Positive
        public InputStream openInputStream() throws DocFileIOException;

    @Positive
        @Override
    @Positive
        public OutputStream openOutputStream() throws DocFileIOException;

    @Positive
        @Override
    @Positive
        public Writer openWriter() throws DocFileIOException, UnsupportedEncodingException;

    @Positive
        @Override
    @Positive
        public boolean canRead();

    @Positive
        @Override
    @Positive
        public boolean canWrite();

    @Positive
        @Override
    @Positive
        public boolean exists();

    @Positive
        @Override
    @Positive
        public String getName();

    @Positive
        @Override
    @Positive
        public String getPath();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean isAbsolute();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean isDirectory();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean isFile();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean isSameFile(DocFile other);

    @Positive
        @Override
    @Positive
        public Iterable<DocFile> list() throws DocFileIOException;

    @Positive
        @Override
    @Positive
        public boolean mkdirs();

    @Positive
        @Override
    @Positive
        public DocFile resolve(DocPath p);

    @Positive
        @Override
    @Positive
        public DocFile resolve(String p);

    @Positive
        @Override
    @Positive
        public DocFile resolveAgainst(Location locn);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
