/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

import org.checkerframework.checker.mustcall.qual.MustCall;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.tainting.qual.Untainted;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.File;
import java.io.FileDescriptor;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;
import jdk.internal.event.ProcessStartEvent;
import sun.security.action.GetPropertyAction;

@AnnotatedFor({ "nullness", "tainting" })
public final class ProcessBuilder {

    public ProcessBuilder(List<@Untainted String> command) {
    }

    public ProcessBuilder(@Untainted String... command) {
    }

    public ProcessBuilder command(List<@Untainted String> command);

    public ProcessBuilder command(@Untainted String... command);

    public List<@Untainted String> command();

    public Map<String, String> environment();

    ProcessBuilder environment(String[] envp);

    @Nullable
    public File directory();

    public ProcessBuilder directory(@Nullable File directory);

    @MustCall()
    static class NullInputStream extends InputStream {

        public int read();

        public int available();
    }

    @MustCall()
    static class NullOutputStream extends OutputStream {

        public void write(int b) throws IOException;
    }

    public abstract static class Redirect {

        public enum Type {

            PIPE, INHERIT, READ, WRITE, APPEND
        }

        public abstract Type type();

        public static final Redirect PIPE;

        public static final Redirect INHERIT;

        public static final Redirect DISCARD;

        public File file();

        boolean append();

        public static Redirect from(final File file);

        public static Redirect to(final File file);

        public static Redirect appendTo(final File file);

        public boolean equals(Object obj);

        public int hashCode();
    }

    static class RedirectPipeImpl extends Redirect {

        @Override
        public Type type();

        @Override
        public String toString();

        FileDescriptor getFd();
    }

    public ProcessBuilder redirectInput(Redirect source);

    public ProcessBuilder redirectOutput(Redirect destination);

    public ProcessBuilder redirectError(Redirect destination);

    public ProcessBuilder redirectInput(File file);

    public ProcessBuilder redirectOutput(File file);

    public ProcessBuilder redirectError(File file);

    public Redirect redirectInput();

    public Redirect redirectOutput();

    public Redirect redirectError();

    public ProcessBuilder inheritIO();

    public boolean redirectErrorStream();

    public ProcessBuilder redirectErrorStream(boolean redirectErrorStream);

    public Process start() throws IOException;

    public static List<Process> startPipeline(List<ProcessBuilder> builders) throws IOException;
}
