/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import org.checkerframework.checker.mustcall.qual.NotOwning;
    @Positive
import jdk.internal.util.StaticProperty;
    @Positive
import java.io.*;
    @Positive
import java.lang.ProcessBuilder.Redirect;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.UnsupportedCharsetException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.concurrent.CompletableFuture;
    @Positive
import java.util.concurrent.ForkJoinPool;
    @Positive
import java.util.concurrent.TimeUnit;
    @Positive
import java.util.stream.Stream;

    @Positive
@AnnotatedFor({ "interning", "nullness", "mustcall" })
    @Positive
@UsesObjectEquals
    @Positive
public abstract class Process {

    @Positive
    public Process() {
    @Positive
    }

    @Positive
    @CFComment({ "nullness: These three methods return @NonNull values despite being documented as", "possibly returning a \"null stream\".  A \"null stream\" is a non-null", "Stream with particular behavior, not a @Nullable Stream reference." })
    @Positive
    @SideEffectFree
    @Positive
    @NotOwning
    @Positive
    public abstract OutputStream getOutputStream();

    @Positive
    @SideEffectFree
    @Positive
    @NotOwning
    @Positive
    public abstract InputStream getInputStream();

    @Positive
    @SideEffectFree
    @Positive
    @NotOwning
    @Positive
    public abstract InputStream getErrorStream();

    @Positive
    public final BufferedReader inputReader();

    @Positive
    public final BufferedReader inputReader(Charset charset);

    @Positive
    public final BufferedReader errorReader();

    @Positive
    public final BufferedReader errorReader(Charset charset);

    @Positive
    public final BufferedWriter outputWriter();

    @Positive
    public final BufferedWriter outputWriter(Charset charset);

    @Positive
    public abstract int waitFor() throws InterruptedException;

    @Positive
    public boolean waitFor(long timeout, TimeUnit unit) throws InterruptedException;

    @Positive
    public abstract int exitValue();

    @Positive
    public abstract void destroy();

    @Positive
    public Process destroyForcibly();

    @Positive
    public boolean supportsNormalTermination();

    @Positive
    public boolean isAlive();

    @Positive
    public long pid();

    @Positive
    public CompletableFuture<Process> onExit();

    @Positive
    public ProcessHandle toHandle();

    @Positive
    public ProcessHandle.Info info();

    @Positive
    public Stream<ProcessHandle> children();

    @Positive
    public Stream<ProcessHandle> descendants();

    @Positive
    static class PipeInputStream extends FileInputStream {

    @Positive
        @Override
    @Positive
        public long skip(long n) throws IOException;
    @Positive
    }

    @Positive
    private static class CharsetHolder {

    @Positive
        static Charset nativeCharset();
    @Positive
    }
    @Positive
}
