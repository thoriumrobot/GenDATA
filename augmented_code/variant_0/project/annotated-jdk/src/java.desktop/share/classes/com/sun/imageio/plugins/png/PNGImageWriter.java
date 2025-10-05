/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.imageio.plugins.png;

    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.image.IndexColorModel;
    @Positive
import java.awt.image.Raster;
    @Positive
import java.awt.image.WritableRaster;
    @Positive
import java.awt.image.RenderedImage;
    @Positive
import java.awt.image.SampleModel;
    @Positive
import java.io.ByteArrayOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Locale;
    @Positive
import java.util.zip.Deflater;
    @Positive
import java.util.zip.DeflaterOutputStream;
    @Positive
import javax.imageio.IIOException;
    @Positive
import javax.imageio.IIOImage;
    @Positive
import javax.imageio.ImageTypeSpecifier;
    @Positive
import javax.imageio.ImageWriteParam;
    @Positive
import javax.imageio.ImageWriter;
    @Positive
import javax.imageio.metadata.IIOMetadata;
    @Positive
import javax.imageio.spi.ImageWriterSpi;
    @Positive
import javax.imageio.stream.ImageOutputStream;
    @Positive
import javax.imageio.stream.ImageOutputStreamImpl;

    @Positive
final class CRC {

    @Positive
    void reset();

    @Positive
    void update(byte[] data, int off, int len);

    @Positive
    void update(int data);

    @Positive
    int getValue();
    @Positive
}

    @Positive
final class ChunkStream extends ImageOutputStreamImpl {

    @Positive
    @Override
    @Positive
    public int read() throws IOException;

    @Positive
    @Override
    @Positive
    public int read(byte[] b, int off, int len) throws IOException;

    @Positive
    @Override
    @Positive
    public void write(@PolySigned byte[] b, int off, int len) throws IOException;

    @Positive
    @Override
    @Positive
    public void write(int b) throws IOException;

    @Positive
    void finish() throws IOException;

    @Positive
    @Override
    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    protected void finalize() throws Throwable;
    @Positive
}

    @Positive
final class IDATOutputStream extends ImageOutputStreamImpl {

    @Positive
    @Override
    @Positive
    public int read() throws IOException;

    @Positive
    @Override
    @Positive
    public int read(byte[] b, int off, int len) throws IOException;

    @Positive
    @Override
    @Positive
    public void write(@PolySigned byte[] b, int off, int len) throws IOException;

    @Positive
    void deflate() throws IOException;

    @Positive
    @Override
    @Positive
    public void write(int b) throws IOException;

    @Positive
    void finish() throws IOException;

    @Positive
    @Override
    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    protected void finalize() throws Throwable;
    @Positive
}

    @Positive
final class PNGImageWriteParam extends ImageWriteParam {

    @Positive
    @Override
    @Positive
    public void unsetCompression();

    @Positive
    @Override
    @Positive
    public boolean isCompressionLossless();

    @Positive
    @Override
    @Positive
    public String[] getCompressionQualityDescriptions();

    @Positive
    @Override
    @Positive
    public float[] getCompressionQualityValues();
    @Positive
}

    @Positive
public final class PNGImageWriter extends ImageWriter {

    @Positive
    public PNGImageWriter(ImageWriterSpi originatingProvider) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void setOutput(Object output);

    @Positive
    @Override
    @Positive
    public ImageWriteParam getDefaultWriteParam();

    @Positive
    @Override
    @Positive
    public IIOMetadata getDefaultStreamMetadata(ImageWriteParam param);

    @Positive
    @Override
    @Positive
    public IIOMetadata getDefaultImageMetadata(ImageTypeSpecifier imageType, ImageWriteParam param);

    @Positive
    @Override
    @Positive
    public IIOMetadata convertStreamMetadata(IIOMetadata inData, ImageWriteParam param);

    @Positive
    @Override
    @Positive
    public IIOMetadata convertImageMetadata(IIOMetadata inData, ImageTypeSpecifier imageType, ImageWriteParam param);

    @Positive
    @Override
    @Positive
    public void write(IIOMetadata streamMetadata, IIOImage image, ImageWriteParam param) throws IIOException;
    @Positive
}
