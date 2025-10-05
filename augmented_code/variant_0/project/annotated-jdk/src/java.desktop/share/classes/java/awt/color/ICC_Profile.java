/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt.color;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.BufferedInputStream;
    @Positive
import java.io.File;
    @Positive
import java.io.FileInputStream;
    @Positive
import java.io.FileOutputStream;
    @Positive
import java.io.FilePermission;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamException;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.StringTokenizer;
    @Positive
import sun.java2d.cmm.CMSManager;
    @Positive
import sun.java2d.cmm.PCMM;
    @Positive
import sun.java2d.cmm.Profile;
    @Positive
import sun.java2d.cmm.ProfileDataVerifier;
    @Positive
import sun.java2d.cmm.ProfileDeferralInfo;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class ICC_Profile implements Serializable {

    @Positive
    private interface BuiltInProfile {

    @Positive
        ICC_Profile SRGB;

    @Positive
        ICC_Profile LRGB;

    @Positive
        ICC_Profile XYZ;

    @Positive
        ICC_Profile PYCC;

    @Positive
        ICC_Profile GRAY;
    @Positive
    }

    @Positive
    public static final int CLASS_INPUT;

    @Positive
    public static final int CLASS_DISPLAY;

    @Positive
    public static final int CLASS_OUTPUT;

    @Positive
    public static final int CLASS_DEVICELINK;

    @Positive
    public static final int CLASS_COLORSPACECONVERSION;

    @Positive
    public static final int CLASS_ABSTRACT;

    @Positive
    public static final int CLASS_NAMEDCOLOR;

    @Positive
    public static final int icSigXYZData;

    @Positive
    public static final int icSigLabData;

    @Positive
    public static final int icSigLuvData;

    @Positive
    public static final int icSigYCbCrData;

    @Positive
    public static final int icSigYxyData;

    @Positive
    public static final int icSigRgbData;

    @Positive
    public static final int icSigGrayData;

    @Positive
    public static final int icSigHsvData;

    @Positive
    public static final int icSigHlsData;

    @Positive
    public static final int icSigCmykData;

    @Positive
    public static final int icSigCmyData;

    @Positive
    public static final int icSigSpace2CLR;

    @Positive
    public static final int icSigSpace3CLR;

    @Positive
    public static final int icSigSpace4CLR;

    @Positive
    public static final int icSigSpace5CLR;

    @Positive
    public static final int icSigSpace6CLR;

    @Positive
    public static final int icSigSpace7CLR;

    @Positive
    public static final int icSigSpace8CLR;

    @Positive
    public static final int icSigSpace9CLR;

    @Positive
    public static final int icSigSpaceACLR;

    @Positive
    public static final int icSigSpaceBCLR;

    @Positive
    public static final int icSigSpaceCCLR;

    @Positive
    public static final int icSigSpaceDCLR;

    @Positive
    public static final int icSigSpaceECLR;

    @Positive
    public static final int icSigSpaceFCLR;

    @Positive
    public static final int icSigInputClass;

    @Positive
    public static final int icSigDisplayClass;

    @Positive
    public static final int icSigOutputClass;

    @Positive
    public static final int icSigLinkClass;

    @Positive
    public static final int icSigAbstractClass;

    @Positive
    public static final int icSigColorSpaceClass;

    @Positive
    public static final int icSigNamedColorClass;

    @Positive
    public static final int icPerceptual;

    @Positive
    public static final int icRelativeColorimetric;

    @Positive
    public static final int icMediaRelativeColorimetric;

    @Positive
    public static final int icSaturation;

    @Positive
    public static final int icAbsoluteColorimetric;

    @Positive
    public static final int icICCAbsoluteColorimetric;

    @Positive
    public static final int icSigHead;

    @Positive
    public static final int icSigAToB0Tag;

    @Positive
    public static final int icSigAToB1Tag;

    @Positive
    public static final int icSigAToB2Tag;

    @Positive
    public static final int icSigBlueColorantTag;

    @Positive
    public static final int icSigBlueMatrixColumnTag;

    @Positive
    public static final int icSigBlueTRCTag;

    @Positive
    public static final int icSigBToA0Tag;

    @Positive
    public static final int icSigBToA1Tag;

    @Positive
    public static final int icSigBToA2Tag;

    @Positive
    public static final int icSigCalibrationDateTimeTag;

    @Positive
    public static final int icSigCharTargetTag;

    @Positive
    public static final int icSigCopyrightTag;

    @Positive
    public static final int icSigCrdInfoTag;

    @Positive
    public static final int icSigDeviceMfgDescTag;

    @Positive
    public static final int icSigDeviceModelDescTag;

    @Positive
    public static final int icSigDeviceSettingsTag;

    @Positive
    public static final int icSigGamutTag;

    @Positive
    public static final int icSigGrayTRCTag;

    @Positive
    public static final int icSigGreenColorantTag;

    @Positive
    public static final int icSigGreenMatrixColumnTag;

    @Positive
    public static final int icSigGreenTRCTag;

    @Positive
    public static final int icSigLuminanceTag;

    @Positive
    public static final int icSigMeasurementTag;

    @Positive
    public static final int icSigMediaBlackPointTag;

    @Positive
    public static final int icSigMediaWhitePointTag;

    @Positive
    public static final int icSigNamedColor2Tag;

    @Positive
    public static final int icSigOutputResponseTag;

    @Positive
    public static final int icSigPreview0Tag;

    @Positive
    public static final int icSigPreview1Tag;

    @Positive
    public static final int icSigPreview2Tag;

    @Positive
    public static final int icSigProfileDescriptionTag;

    @Positive
    public static final int icSigProfileSequenceDescTag;

    @Positive
    public static final int icSigPs2CRD0Tag;

    @Positive
    public static final int icSigPs2CRD1Tag;

    @Positive
    public static final int icSigPs2CRD2Tag;

    @Positive
    public static final int icSigPs2CRD3Tag;

    @Positive
    public static final int icSigPs2CSATag;

    @Positive
    public static final int icSigPs2RenderingIntentTag;

    @Positive
    public static final int icSigRedColorantTag;

    @Positive
    public static final int icSigRedMatrixColumnTag;

    @Positive
    public static final int icSigRedTRCTag;

    @Positive
    public static final int icSigScreeningDescTag;

    @Positive
    public static final int icSigScreeningTag;

    @Positive
    public static final int icSigTechnologyTag;

    @Positive
    public static final int icSigUcrBgTag;

    @Positive
    public static final int icSigViewingCondDescTag;

    @Positive
    public static final int icSigViewingConditionsTag;

    @Positive
    public static final int icSigChromaticityTag;

    @Positive
    public static final int icSigChromaticAdaptationTag;

    @Positive
    public static final int icSigColorantOrderTag;

    @Positive
    public static final int icSigColorantTableTag;

    @Positive
    public static final int icHdrSize;

    @Positive
    public static final int icHdrCmmId;

    @Positive
    public static final int icHdrVersion;

    @Positive
    public static final int icHdrDeviceClass;

    @Positive
    public static final int icHdrColorSpace;

    @Positive
    public static final int icHdrPcs;

    @Positive
    public static final int icHdrDate;

    @Positive
    public static final int icHdrMagic;

    @Positive
    public static final int icHdrPlatform;

    @Positive
    public static final int icHdrFlags;

    @Positive
    public static final int icHdrManufacturer;

    @Positive
    public static final int icHdrModel;

    @Positive
    public static final int icHdrAttributes;

    @Positive
    public static final int icHdrRenderingIntent;

    @Positive
    public static final int icHdrIlluminant;

    @Positive
    public static final int icHdrCreator;

    @Positive
    public static final int icHdrProfileID;

    @Positive
    public static final int icTagType;

    @Positive
    public static final int icTagReserved;

    @Positive
    public static final int icCurveCount;

    @Positive
    public static final int icCurveData;

    @Positive
    public static final int icXYZNumberX;

    @Positive
    @Deprecated()
    @Positive
    @SuppressWarnings("removal")
    @Positive
    protected void finalize();

    @Positive
    public static ICC_Profile getInstance(byte[] data);

    @Positive
    public static ICC_Profile getInstance(int cspace);

    @Positive
    public static ICC_Profile getInstance(String fileName) throws IOException;

    @Positive
    public static ICC_Profile getInstance(InputStream s) throws IOException;

    @Positive
    static byte[] getProfileDataFromStream(InputStream s) throws IOException;

    @Positive
    public int getMajorVersion();

    @Positive
    public int getMinorVersion();

    @Positive
    public int getProfileClass();

    @Positive
    public int getColorSpaceType();

    @Positive
    public int getPCSType();

    @Positive
    public void write(String fileName) throws IOException;

    @Positive
    public void write(OutputStream s) throws IOException;

    @Positive
    public byte[] getData();

    @Positive
    public byte[] getData(int tagSignature);

    @Positive
    public void setData(int tagSignature, byte[] tagData);

    @Positive
    public int getNumComponents();

    @Positive
    float[] getMediaWhitePoint();

    @Positive
    final float[] getXYZTag(int tagSignature);

    @Positive
    float getGamma(int tagSignature);

    @Positive
    short[] getTRC(int tagSignature);

    @Positive
    @Serial
    @Positive
    protected Object readResolve() throws ObjectStreamException;
    @Positive
}
