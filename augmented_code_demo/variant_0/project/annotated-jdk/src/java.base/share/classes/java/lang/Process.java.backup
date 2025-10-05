/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import org.checkerframework.checker.mustcall.qual.NotOwning;
import jdk.internal.util.StaticProperty;
import java.io.*;
import java.lang.ProcessBuilder.Redirect;
import java.nio.charset.Charset;
import java.nio.charset.UnsupportedCharsetException;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

@AnnotatedFor({ "interning", "nullness", "mustcall" })
@UsesObjectEquals
public abstract class Process {

    public Process() {
    }

    @CFComment({ "nullness: These three methods return @NonNull values despite being documented as", "possibly returning a \"null stream\".  A \"null stream\" is a non-null", "Stream with particular behavior, not a @Nullable Stream reference." })
    @SideEffectFree
    @NotOwning
    public abstract OutputStream getOutputStream();

    @SideEffectFree
    @NotOwning
    public abstract InputStream getInputStream();

    @SideEffectFree
    @NotOwning
    public abstract InputStream getErrorStream();

    public final BufferedReader inputReader();

    public final BufferedReader inputReader(Charset charset);

    public final BufferedReader errorReader();

    public final BufferedReader errorReader(Charset charset);

    public final BufferedWriter outputWriter();

    public final BufferedWriter outputWriter(Charset charset);

    public abstract int waitFor() throws InterruptedException;

    public boolean waitFor(long timeout, TimeUnit unit) throws InterruptedException;

    public abstract int exitValue();

    public abstract void destroy();

    public Process destroyForcibly();

    public boolean supportsNormalTermination();

    public boolean isAlive();

    public long pid();

    public CompletableFuture<Process> onExit();

    public ProcessHandle toHandle();

    public ProcessHandle.Info info();

    public Stream<ProcessHandle> children();

    public Stream<ProcessHandle> descendants();

    static class PipeInputStream extends FileInputStream {

        @Override
        public long skip(long n) throws IOException;
    }

    private static class CharsetHolder {

        static Charset nativeCharset();
    }
}
