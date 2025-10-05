/*
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
package org.w3c.dom.ls;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public interface LSInput {

    @Positive
    @Pure
    @Positive
    public java.io.@Nullable Reader getCharacterStream();

    @Positive
    public void setCharacterStream(java.io.@Nullable Reader characterStream);

    @Positive
    @Pure
    @Positive
    public java.io.@Nullable InputStream getByteStream();

    @Positive
    public void setByteStream(java.io.@Nullable InputStream byteStream);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getStringData();

    @Positive
    public void setStringData(@Nullable String stringData);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getSystemId();

    @Positive
    public void setSystemId(@Nullable String systemId);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getPublicId();

    @Positive
    public void setPublicId(@Nullable String publicId);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getBaseURI();

    @Positive
    public void setBaseURI(@Nullable String baseURI);

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public String getEncoding();

    @Positive
    public void setEncoding(@Nullable String encoding);

    @Positive
    @Pure
    @Positive
    public boolean getCertifiedText();

    @Positive
    public void setCertifiedText(boolean certifiedText);
    @Positive
}

// CFWR semantic augmentation - variant 1
