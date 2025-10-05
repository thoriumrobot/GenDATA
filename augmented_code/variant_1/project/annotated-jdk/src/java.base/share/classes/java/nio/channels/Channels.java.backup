/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.channels;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Reader;
import java.io.Writer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.UnsupportedCharsetException;
import java.nio.channels.spi.AbstractInterruptibleChannel;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import sun.nio.ch.ChannelInputStream;
import sun.nio.cs.StreamDecoder;
import sun.nio.cs.StreamEncoder;

@AnnotatedFor({ "interning", "mustcall" })
@UsesObjectEquals
public final class Channels {

    @MustCallAlias
    public static InputStream newInputStream(@MustCallAlias ReadableByteChannel ch);

    @MustCallAlias
    public static OutputStream newOutputStream(@MustCallAlias WritableByteChannel ch);

    @MustCallAlias
    public static InputStream newInputStream(@MustCallAlias AsynchronousByteChannel ch);

    @MustCallAlias
    public static OutputStream newOutputStream(@MustCallAlias AsynchronousByteChannel ch);

    @MustCallAlias
    public static ReadableByteChannel newChannel(@MustCallAlias InputStream in);

    private static class ReadableByteChannelImpl extends AbstractInterruptibleChannel implements ReadableByteChannel {

        @Override
        public int read(ByteBuffer dst) throws IOException;

        @Override
        protected void implCloseChannel() throws IOException;
    }

    @MustCallAlias
    public static WritableByteChannel newChannel(@MustCallAlias OutputStream out);

    private static class WritableByteChannelImpl extends AbstractInterruptibleChannel implements WritableByteChannel {

        @Override
        public int write(ByteBuffer src) throws IOException;

        @Override
        protected void implCloseChannel() throws IOException;
    }

    @MustCallAlias
    public static Reader newReader(@MustCallAlias ReadableByteChannel ch, CharsetDecoder dec, int minBufferCap);

    @MustCallAlias
    public static Reader newReader(@MustCallAlias ReadableByteChannel ch, String csName);

    @MustCallAlias
    public static Reader newReader(@MustCallAlias ReadableByteChannel ch, Charset charset);

    @MustCallAlias
    public static Writer newWriter(@MustCallAlias WritableByteChannel ch, CharsetEncoder enc, int minBufferCap);

    @MustCallAlias
    public static Writer newWriter(@MustCallAlias WritableByteChannel ch, String csName);

    @MustCallAlias
    public static Writer newWriter(@MustCallAlias WritableByteChannel ch, Charset charset);
}
