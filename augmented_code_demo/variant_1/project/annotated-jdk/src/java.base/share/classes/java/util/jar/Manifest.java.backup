/*
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util.jar;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.DataOutputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;
import sun.nio.cs.UTF_8;
import sun.security.util.SecurityProperties;

public class Manifest implements Cloneable {

    public Manifest() {
    }

    public Manifest(InputStream is) throws IOException {
    }

    public Manifest(Manifest man) {
    }

    public Attributes getMainAttributes();

    public Map<String, Attributes> getEntries();

    public Attributes getAttributes(String name);

    Attributes getTrustedAttributes(String name);

    public void clear();

    public void write(OutputStream out) throws IOException;

    @Deprecated()
    static void make72Safe(StringBuffer line);

    static void println72(OutputStream out, String line) throws IOException;

    static void println(OutputStream out) throws IOException;

    static String getErrorPosition(String filename, final int lineNumber);

    public void read(InputStream is) throws IOException;

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object o);

    public int hashCode();

    public Object clone();

    static class FastInputStream extends FilterInputStream {

        public int read() throws IOException;

        public int read(byte[] b, int off, int len) throws IOException;

        public int readLine(byte[] b, int off, int len) throws IOException;

        @Pure
        public byte peek() throws IOException;

        public int readLine(byte[] b) throws IOException;

        public long skip(long n) throws IOException;

        public int available() throws IOException;

        public void close() throws IOException;
    }
}
