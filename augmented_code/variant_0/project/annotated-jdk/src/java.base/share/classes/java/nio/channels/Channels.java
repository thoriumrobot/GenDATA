/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.channels;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.FileInputStream;
    @Positive
import java.io.FileOutputStream;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.Reader;
    @Positive
import java.io.Writer;
    @Positive
import java.io.IOException;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.CharsetDecoder;
    @Positive
import java.nio.charset.CharsetEncoder;
    @Positive
import java.nio.charset.UnsupportedCharsetException;
    @Positive
import java.nio.channels.spi.AbstractInterruptibleChannel;
    @Positive
import java.util.Objects;
    @Positive
import java.util.concurrent.ExecutionException;
    @Positive
import sun.nio.ch.ChannelInputStream;
    @Positive
import sun.nio.cs.StreamDecoder;
    @Positive
import sun.nio.cs.StreamEncoder;

    @Positive
@AnnotatedFor({ "interning", "mustcall" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Channels {

    @Positive
    @MustCallAlias
    @Positive
    public static InputStream newInputStream(@MustCallAlias ReadableByteChannel ch);

    @Positive
    @MustCallAlias
    @Positive
    public static OutputStream newOutputStream(@MustCallAlias WritableByteChannel ch);

    @Positive
    @MustCallAlias
    @Positive
    public static InputStream newInputStream(@MustCallAlias AsynchronousByteChannel ch);

    @Positive
    @MustCallAlias
    @Positive
    public static OutputStream newOutputStream(@MustCallAlias AsynchronousByteChannel ch);

    @Positive
    @MustCallAlias
    @Positive
    public static ReadableByteChannel newChannel(@MustCallAlias InputStream in);

    @Positive
    private static class ReadableByteChannelImpl extends AbstractInterruptibleChannel implements ReadableByteChannel {

    @Positive
        @Override
    @Positive
        public int read(ByteBuffer dst) throws IOException;

    @Positive
        @Override
    @Positive
        protected void implCloseChannel() throws IOException;
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public static WritableByteChannel newChannel(@MustCallAlias OutputStream out);

    @Positive
    private static class WritableByteChannelImpl extends AbstractInterruptibleChannel implements WritableByteChannel {

    @Positive
        @Override
    @Positive
        public int write(ByteBuffer src) throws IOException;

    @Positive
        @Override
    @Positive
        protected void implCloseChannel() throws IOException;
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public static Reader newReader(@MustCallAlias ReadableByteChannel ch, CharsetDecoder dec, int minBufferCap);

    @Positive
    @MustCallAlias
    @Positive
    public static Reader newReader(@MustCallAlias ReadableByteChannel ch, String csName);

    @Positive
    @MustCallAlias
    @Positive
    public static Reader newReader(@MustCallAlias ReadableByteChannel ch, Charset charset);

    @Positive
    @MustCallAlias
    @Positive
    public static Writer newWriter(@MustCallAlias WritableByteChannel ch, CharsetEncoder enc, int minBufferCap);

    @Positive
    @MustCallAlias
    @Positive
    public static Writer newWriter(@MustCallAlias WritableByteChannel ch, String csName);

    @Positive
    @MustCallAlias
    @Positive
    public static Writer newWriter(@MustCallAlias WritableByteChannel ch, Charset charset);
    @Positive
}
