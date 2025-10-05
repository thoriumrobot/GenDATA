/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util.jar;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.io.DataOutputStream;
    @Positive
import java.io.FilterInputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Map;
    @Positive
import sun.nio.cs.UTF_8;
    @Positive
import sun.security.util.SecurityProperties;

    @Positive
public class Manifest implements Cloneable {

    @Positive
    public Manifest() {
    @Positive
    }

    @Positive
    public Manifest(InputStream is) throws IOException {
    @Positive
    }

    @Positive
    public Manifest(Manifest man) {
    @Positive
    }

    @Positive
    public Attributes getMainAttributes();

    @Positive
    public Map<String, Attributes> getEntries();

    @Positive
    public Attributes getAttributes(String name);

    @Positive
    Attributes getTrustedAttributes(String name);

    @Positive
    public void clear();

    @Positive
    public void write(OutputStream out) throws IOException;

    @Positive
    @Deprecated()
    @Positive
    static void make72Safe(StringBuffer line);

    @Positive
    static void println72(OutputStream out, String line) throws IOException;

    @Positive
    static void println(OutputStream out) throws IOException;

    @Positive
    static String getErrorPosition(String filename, final int lineNumber);

    @Positive
    public void read(InputStream is) throws IOException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    public int hashCode();

    @Positive
    public Object clone();

    @Positive
    static class FastInputStream extends FilterInputStream {

    @Positive
        public int read() throws IOException;

    @Positive
        public int read(byte[] b, int off, int len) throws IOException;

    @Positive
        public int readLine(byte[] b, int off, int len) throws IOException;

    @Positive
        @Pure
    @Positive
        public byte peek() throws IOException;

    @Positive
        public int readLine(byte[] b) throws IOException;

    @Positive
        public long skip(long n) throws IOException;

    @Positive
        public int available() throws IOException;

    @Positive
        public void close() throws IOException;
    @Positive
    }
    @Positive
}
