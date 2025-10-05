/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.mustcall.qual.MustCall;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.tainting.qual.Untainted;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.File;
    @Positive
import java.io.FileDescriptor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.StringJoiner;
    @Positive
import jdk.internal.event.ProcessStartEvent;
    @Positive
import sun.security.action.GetPropertyAction;

    @Positive
@AnnotatedFor({ "nullness", "tainting" })
    @Positive
public final class ProcessBuilder {

    @Positive
    public ProcessBuilder(List<@Untainted String> command) {
    @Positive
    }

    @Positive
    public ProcessBuilder(@Untainted String... command) {
    @Positive
    }

    @Positive
    public ProcessBuilder command(List<@Untainted String> command);

    @Positive
    public ProcessBuilder command(@Untainted String... command);

    @Positive
    public List<@Untainted String> command();

    @Positive
    public Map<String, String> environment();

    @Positive
    ProcessBuilder environment(String[] envp);

    @Positive
    @Nullable
    @Positive
    public File directory();

    @Positive
    public ProcessBuilder directory(@Nullable File directory);

    @Positive
    @MustCall()
    @Positive
    static class NullInputStream extends InputStream {

    @Positive
        public int read();

    @Positive
        public int available();
    @Positive
    }

    @Positive
    @MustCall()
    @Positive
    static class NullOutputStream extends OutputStream {

    @Positive
        public void write(int b) throws IOException;
    @Positive
    }

    @Positive
    public abstract static class Redirect {

    @Positive
        public enum Type {

    @Positive
            PIPE, INHERIT, READ, WRITE, APPEND
    @Positive
        }

    @Positive
        public abstract Type type();

    @Positive
        public static final Redirect PIPE;

    @Positive
        public static final Redirect INHERIT;

    @Positive
        public static final Redirect DISCARD;

    @Positive
        public File file();

    @Positive
        boolean append();

    @Positive
        public static Redirect from(final File file);

    @Positive
        public static Redirect to(final File file);

    @Positive
        public static Redirect appendTo(final File file);

    @Positive
        public boolean equals(Object obj);

    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    static class RedirectPipeImpl extends Redirect {

    @Positive
        @Override
    @Positive
        public Type type();

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        FileDescriptor getFd();
    @Positive
    }

    @Positive
    public ProcessBuilder redirectInput(Redirect source);

    @Positive
    public ProcessBuilder redirectOutput(Redirect destination);

    @Positive
    public ProcessBuilder redirectError(Redirect destination);

    @Positive
    public ProcessBuilder redirectInput(File file);

    @Positive
    public ProcessBuilder redirectOutput(File file);

    @Positive
    public ProcessBuilder redirectError(File file);

    @Positive
    public Redirect redirectInput();

    @Positive
    public Redirect redirectOutput();

    @Positive
    public Redirect redirectError();

    @Positive
    public ProcessBuilder inheritIO();

    @Positive
    public boolean redirectErrorStream();

    @Positive
    public ProcessBuilder redirectErrorStream(boolean redirectErrorStream);

    @Positive
    public Process start() throws IOException;

    @Positive
    public static List<Process> startPipeline(List<ProcessBuilder> builders) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
