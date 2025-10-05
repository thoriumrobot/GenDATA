/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
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
