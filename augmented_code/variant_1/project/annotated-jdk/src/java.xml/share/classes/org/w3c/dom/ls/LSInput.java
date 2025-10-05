/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
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
