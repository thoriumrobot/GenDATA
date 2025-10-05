/*
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.imageio.plugins.png;

import org.checkerframework.checker.signedness.qual.PolySigned;
import java.awt.Rectangle;
import java.awt.image.IndexColorModel;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.awt.image.RenderedImage;
import java.awt.image.SampleModel;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Iterator;
import java.util.Locale;
import java.util.zip.Deflater;
import java.util.zip.DeflaterOutputStream;
import javax.imageio.IIOException;
import javax.imageio.IIOImage;
import javax.imageio.ImageTypeSpecifier;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.spi.ImageWriterSpi;
import javax.imageio.stream.ImageOutputStream;
import javax.imageio.stream.ImageOutputStreamImpl;

final class CRC {

    void reset();

    void update(byte[] data, int off, int len);

    void update(int data);

    int getValue();
}

final class ChunkStream extends ImageOutputStreamImpl {

    @Override
    public int read() throws IOException;

    @Override
    public int read(byte[] b, int off, int len) throws IOException;

    @Override
    public void write(@PolySigned byte[] b, int off, int len) throws IOException;

    @Override
    public void write(int b) throws IOException;

    void finish() throws IOException;

    @Override
    @SuppressWarnings("deprecation")
    protected void finalize() throws Throwable;
}

final class IDATOutputStream extends ImageOutputStreamImpl {

    @Override
    public int read() throws IOException;

    @Override
    public int read(byte[] b, int off, int len) throws IOException;

    @Override
    public void write(@PolySigned byte[] b, int off, int len) throws IOException;

    void deflate() throws IOException;

    @Override
    public void write(int b) throws IOException;

    void finish() throws IOException;

    @Override
    @SuppressWarnings("deprecation")
    protected void finalize() throws Throwable;
}

final class PNGImageWriteParam extends ImageWriteParam {

    @Override
    public void unsetCompression();

    @Override
    public boolean isCompressionLossless();

    @Override
    public String[] getCompressionQualityDescriptions();

    @Override
    public float[] getCompressionQualityValues();
}

public final class PNGImageWriter extends ImageWriter {

    public PNGImageWriter(ImageWriterSpi originatingProvider) {
    }

    @Override
    public void setOutput(Object output);

    @Override
    public ImageWriteParam getDefaultWriteParam();

    @Override
    public IIOMetadata getDefaultStreamMetadata(ImageWriteParam param);

    @Override
    public IIOMetadata getDefaultImageMetadata(ImageTypeSpecifier imageType, ImageWriteParam param);

    @Override
    public IIOMetadata convertStreamMetadata(IIOMetadata inData, ImageWriteParam param);

    @Override
    public IIOMetadata convertImageMetadata(IIOMetadata inData, ImageTypeSpecifier imageType, ImageWriteParam param);

    @Override
    public void write(IIOMetadata streamMetadata, IIOImage image, ImageWriteParam param) throws IIOException;
}
